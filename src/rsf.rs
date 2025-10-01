// src/rsf.rs

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use rand::prelude::IndexedRandom;
use std::collections::BTreeSet;
use ordered_float::OrderedFloat;

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
    time: &Vec<f64>,
    status: &Vec<u8>,
    indices: Vec<usize>,
    min_node_size: usize,
    max_features: usize,
    rng: &mut StdRng,
) -> TreeNode {
    let n_features = data.ncols();
    let mut node = TreeNode::new(indices.clone());

    if indices.len() <= min_node_size {
        return node; // Leaf node
    }

    // Sample features randomly
    let array_indices: Vec<usize> = (0..n_features).collect();
    let sampled_features: Vec<usize> = array_indices
        .choose_multiple(rng, max_features)
        .cloned()
        .collect();

    let mut best_stat = 0.0;
    let mut best_feature = None;
    let mut best_value = None;

    // Find best split
    for &f in &sampled_features {
        // Only consider rows where feature is not NaN
        let vals: Vec<f64> = indices.iter()
            .map(|&i| data[[i, f]])
            .filter(|v| !v.is_nan())
            .collect();

        if vals.is_empty() {
            continue; // Cannot split on this feature
        }

        let median = median(&vals);

        let left_idx: Vec<usize> = indices.iter()
            .cloned()
            .filter(|&i| {
                let val = data[[i, f]];
                !val.is_nan() && val <= median
            })
            .collect();

        let right_idx: Vec<usize> = indices.iter()
            .cloned()
            .filter(|&i| {
                let val = data[[i, f]];
                !val.is_nan() && val > median
            })
            .collect();

        if left_idx.is_empty() || right_idx.is_empty() {
            continue;
        }

        let stat = log_rank_stat(time, status, &left_idx, &right_idx);
        if stat > best_stat {
            best_stat = stat;
            best_feature = Some(f);
            best_value = Some(median);
        }
    }

    // If no valid split, return leaf
    let (f, v) = match (best_feature, best_value) {
        (Some(f), Some(v)) => (f, v),
        _ => return node,
    };

    node.split_feature = Some(f);
    node.split_value = Some(v);

    // Split rows for left/right children
    let left_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| {
            let val = data[[i, f]];
            !val.is_nan() && val <= v
        })
        .collect();

    let right_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| {
            let val = data[[i, f]];
            !val.is_nan() && val > v
        })
        .collect();

    node.left = Some(Box::new(build_tree(data, time, status, left_idx, min_node_size, max_features, rng)));
    node.right = Some(Box::new(build_tree(data, time, status, right_idx, min_node_size, max_features, rng)));

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

/// Fit a full RSF
pub fn fit_rsf(
    data: &Array2<f64>,
    time: &Vec<f64>,
    status: &Vec<u8>,
    config: &RSFConfig,
) -> RSFModel {
    let n_features = data.ncols();
    let max_features = config.max_features.unwrap_or((n_features as f64).sqrt() as usize);

    let mut feature_importance: HashMap<usize, f64> = HashMap::new();

    for f in 0..n_features {
        feature_importance.insert(f, 0.0);
    }


    println!("fitting the survival random forest model");

    // Use rayon's parallel iterator for the trees
    let trees: Vec<TreeNode> = (0..config.n_trees).into_par_iter().map(|_| {
        // Each thread gets its own RNG
        let mut rng = StdRng::seed_from_u64(config.seed);

        // Bootstrap sample
        let indices: Vec<usize> = (0..data.nrows()).collect();
        let sample_indices: Vec<usize> = indices.choose_multiple(&mut rng, data.nrows()).cloned().collect();

        // Build the tree
        build_tree(data, time, status, sample_indices, config.min_node_size, max_features, &mut rng)
    }).collect();

    // Aggregate feature importance after all trees
    for t in &trees {
        increment_importance(&t, &mut feature_importance);
    }

    RSFModel { trees, feature_importance }
}

/// Increment feature importance recursively
fn increment_importance(node: &TreeNode, importance: &mut HashMap<usize,f64>) {
    if let Some(f) = node.split_feature {
        *importance.get_mut(&f).unwrap() += 1.0;
    }
    if let Some(left) = &node.left {
        increment_importance(left, importance);
    }
    if let Some(right) = &node.right {
        increment_importance(right, importance);
    }
}

