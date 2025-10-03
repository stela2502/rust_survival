// src/rsf.rs

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use rand::prelude::IndexedRandom;
use std::collections::BTreeSet;
use ordered_float::OrderedFloat;
use rand::prelude::IteratorRandom;

use rayon::prelude::*;

/// Node of the survival tree
#[derive(Debug)]
#[allow(dead_code)]
pub struct TreeNode {
    split_feature: Option<usize>,
    split_value: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    indices: Vec<usize>, // rows in this node
}

impl TreeNode {
    fn new(indices: Vec<usize>) -> Self {
        TreeNode {
            split_feature: None,
            split_value: None,
            left: None,
            right: None,
            indices,
        }
    }
}

/// RSF configuration
#[allow(dead_code)]
pub struct RSFConfig {
    pub n_trees: usize,
    pub min_node_size: usize,
    pub max_features: Option<usize>,
    pub seed: u64,
}

/// RSF model output
#[allow(dead_code)]
pub struct RSFModel {
    pub trees: Vec<TreeNode>,
    pub feature_importance: HashMap<usize, f64>, // column index -> importance
}


/// Compute log-rank statistic (Mantel-Haenszel version)
pub fn log_rank_stat(
    time: &[f64],
    status: &[u8],
    left_idx: &[usize],
    right_idx: &[usize],
) -> f64 {
    assert_eq!(time.len(), status.len());

    // Collect all unique event times
    let mut event_times: BTreeSet<OrderedFloat<f64>> = BTreeSet::new();
    for (&t, &s) in time.iter().zip(status.iter()) {
        if s == 1 {
            event_times.insert(OrderedFloat(t));
        }
    }

    let mut o_l = 0.0; // observed events left
    let mut e_l = 0.0; // expected events left
    let mut v = 0.0;   // variance



    for &tmp in &event_times {
        // risk sets at time t
        let t = f64::from(tmp);
        let y_l = left_idx.iter().filter(|&&i| time[i] >= t as f64).count() as f64;
        let y_r = right_idx.iter().filter(|&&i| time[i] >= t as f64 ).count() as f64;
        let y = y_l + y_r;
        if y <= 1.0 {
            continue;
        }

        // events at time t
        let d_l = left_idx.iter().filter(|&&i| (time[i] - t).abs() < f64::EPSILON && status[i] == 1).count() as f64;
        let d_r = right_idx.iter().filter(|&&i| (time[i] - t).abs() < f64::EPSILON && status[i] == 1).count() as f64;
        let d = d_l + d_r;
        if d == 0.0 {
            continue;
        }

        // expected and variance
        let e = d * (y_l / y);
        let v_t = (y_l * y_r * d * (y - d)) / (y * y * (y - 1.0));

        o_l += d_l;
        e_l += e;
        v += v_t;
    }

    if v <= 0.0 {
        0.0
    } else {
        (o_l - e_l).abs() / v.sqrt()
    }
}

/// Build a single survival tree recursively allowing NA values
fn build_tree(
    data: &Array2<f64>,
    time: &[f64],
    status: &[u8],
    indices: Vec<usize>,
    min_node_size: usize,
    max_features: usize,
    rng: &mut StdRng,
    feature_importance: &mut HashMap<usize, f64>,
) -> TreeNode {
    let n_features = data.ncols();
    let mut node = TreeNode::new(indices.clone());

    if indices.len() <= min_node_size {
        return node; // Leaf node
    }

    // Sample features randomly
    let sampled_features: Vec<usize> = (0..n_features)
        .collect::<Vec<_>>()
        .choose_multiple(rng, max_features)
        .cloned()
        .collect();

    //println!("Tree got these features: {:?}", &sampled_features);

    let mut best_stat = 0.0;
    let mut best_feature = None;
    let mut best_value = None;

    for &f in &sampled_features {
        // Collect non-NA values
        let vals: Vec<f64> = indices.iter()
            .map(|&i| data[[i, f]])
            .filter(|v| !v.is_nan())
            .collect();

        if vals.len() < 2 {
            continue;
        }

        // Try multiple candidate splits (quantiles)
        let mut candidates = vec![];
        let mut sorted_vals = vals.clone();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for q in [0.25, 0.5, 0.75].iter() {
            let idx = ((*q * (sorted_vals.len() as f64)) as usize).min(sorted_vals.len() - 1);
            candidates.push(sorted_vals[idx]);
        }

        for &split_val in &candidates {
            let left_idx: Vec<usize> = indices.iter()
                .cloned()
                .filter(|&i| {
                    let val = data[[i, f]];
                    val <= split_val || val.is_nan() // NA can go left
                })
                .collect();

            let right_idx: Vec<usize> = indices.iter()
                .cloned()
                .filter(|&i| {
                    let val = data[[i, f]];
                    val > split_val || val.is_nan() // NA can also go right
                })
                .collect();

            if left_idx.is_empty() || right_idx.is_empty() {
                continue;
            }

            let stat = log_rank_stat(time, status, &left_idx, &right_idx);
            if stat > best_stat {
                best_stat = stat;
                best_feature = Some(f);
                best_value = Some(split_val);
            }
        }
    }

    let (f, v) = match (best_feature, best_value) {
        (Some(f), Some(v)) => (f, v),
        _ => return node,
    };

    node.split_feature = Some(f);
    node.split_value = Some(v);

    // Update feature importance weighted by log-rank statistic
    *feature_importance.get_mut(&f).unwrap() += best_stat;

    // Split rows for children
    let left_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| {
            let val = data[[i, f]];
            val <= v || val.is_nan()
        })
        .collect();

    let right_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| {
            let val = data[[i, f]];
            val > v || val.is_nan()
        })
        .collect();

    node.left = Some(Box::new(build_tree(
        data, time, status, left_idx, min_node_size, max_features, rng, feature_importance
    )));
    node.right = Some(Box::new(build_tree(
        data, time, status, right_idx, min_node_size, max_features, rng, feature_importance
    )));

    node
}
/// Median helper
fn median(vals: &Vec<f64>) -> f64 {
    let mut sorted = vals.clone();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n/2 - 1] + sorted[n/2]) / 2.0
    } else {
        sorted[n/2]
    }
}


pub fn fit_rsf(
    data: &Array2<f64>,
    time: &[f64],
    status: &[u8],
    config: &RSFConfig,
) -> RSFModel {
    let n_features = data.ncols();

    let max_auto = std::cmp::max(
        (n_features as f64 * 0.4) as usize,   // 10% of total features
        (n_features as f64).sqrt() as usize   // sqrt heuristic
    );
    let max_features = config.max_features.unwrap_or(max_auto) ;

    println!("Fitting survival random forest with {} trees and {} features per tree...", config.n_trees, max_features);

    // Initialize global feature importance
    let mut feature_importance: HashMap<usize, f64> = (0..n_features)
        .map(|f| (f, 0.0))
        .collect();

    // Build trees in parallel
    let trees: Vec<TreeNode> = (0..config.n_trees).into_par_iter().map(|tree_idx| {
        // Unique RNG per tree
        let mut rng = StdRng::seed_from_u64(config.seed + tree_idx as u64);

        // Bootstrap sample
        let indices: Vec<usize> = (0..data.nrows()).collect();
        let sample_indices: Vec<usize> = indices.choose_multiple(&mut rng, data.nrows()).cloned().collect();

        // Local feature importance for this tree
        let mut tree_importance: HashMap<usize, f64> = (0..n_features).map(|f| (f, 0.0)).collect();

        // Build the tree
        let tree = build_tree(
            data,
            time,
            status,
            sample_indices,
            config.min_node_size,
            max_features,
            &mut rng,
            &mut tree_importance
        );

        // Return both tree and its importance
        (tree, tree_importance)
    }).collect::<Vec<_>>()
    .into_iter()
    .map(|(tree, tree_importance)| {
        // Merge tree importance into global importance (thread-safe since we are done with parallel)
        for (k, v) in tree_importance {
            *feature_importance.get_mut(&k).unwrap() += v;
        }
        tree
    }).collect();

    RSFModel { trees, feature_importance }
}

/// Increment feature importance recursively using absolute log-rank statistic
fn increment_importance(
    node: &TreeNode,
    data: &Array2<f64>,
    time: &[f64],
    status: &[u8],
    importance: &mut HashMap<usize, f64>,
    min_node_importance: usize, // minimum samples per child to count
) {
    if let Some(f) = node.split_feature {
        let split_val = node.split_value.unwrap();

        // Determine left/right indices at this node
        let left_idx: Vec<usize> = node.indices.iter()
            .cloned()
            .filter(|&i| {
                let val = data[[i, f]];
                val <= split_val || val.is_nan()
            })
            .collect();

        let right_idx: Vec<usize> = node.indices.iter()
            .cloned()
            .filter(|&i| {
                let val = data[[i, f]];
                val > split_val || val.is_nan()
            })
            .collect();

        // Only increment importance if both children have enough samples
        if left_idx.len() >= min_node_importance && right_idx.len() >= min_node_importance {
            let stat = log_rank_stat(time, status, &left_idx, &right_idx);
            *importance.get_mut(&f).unwrap() += stat.abs(); // use absolute value to avoid negative importance
        }
    }

    // Recurse into children
    if let Some(left) = &node.left {
        increment_importance(left, data, time, status, importance, min_node_importance);
    }
    if let Some(right) = &node.right {
        increment_importance(right, data, time, status, importance, min_node_importance);
    }
}
