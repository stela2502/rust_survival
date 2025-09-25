// src/rsf.rs

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashMap;

/// Node of the survival tree
#[derive(Debug)]
struct TreeNode {
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
pub struct RSFConfig {
    pub n_trees: usize,
    pub min_node_size: usize,
    pub max_features: Option<usize>,
    pub seed: u64,
}

/// RSF model output
pub struct RSFModel {
    pub trees: Vec<TreeNode>,
    pub feature_importance: HashMap<usize, f64>, // column index -> importance
}

/// Compute log-rank statistic for a split
fn log_rank_stat(
    time: &Vec<f64>,
    status: &Vec<u8>,
    left_idx: &Vec<usize>,
    right_idx: &Vec<usize>,
) -> f64 {
    // Simplified unweighted log-rank statistic
    let mut o_l = 0.0;
    let mut e_l = 0.0;

    for (&idx_l, &idx_r) in left_idx.iter().zip(right_idx.iter()) {
        // basic: sum events in left minus expected (placeholder, more precise can be done)
        o_l += status[idx_l] as f64;
        e_l += (status[idx_l] as f64 + status[idx_r] as f64) / 2.0;
    }

    if e_l == 0.0 { 0.0 } else { (o_l - e_l).abs() }
}

/// Build a single survival tree recursively
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
        return node;
    }

    // Sample features randomly
    let feature_indices: Vec<usize> = (0..n_features).collect();
    let sampled_features: Vec<usize> = feature_indices
        .choose_multiple(rng, max_features)
        .cloned()
        .collect();

    // Find best split using log-rank statistic
    let mut best_stat = 0.0;
    let mut best_feature = None;
    let mut best_value = None;

    for &f in &sampled_features {
        // candidate split: median of the feature in this node
        let vals: Vec<f64> = indices.iter().map(|&i| data[[i, f]]).collect();
        let median = median(&vals);

        let left_idx: Vec<usize> = indices.iter().cloned().filter(|&i| data[[i,f]] <= median).collect();
        let right_idx: Vec<usize> = indices.iter().cloned().filter(|&i| data[[i,f]] > median).collect();

        if left_idx.len() == 0 || right_idx.len() == 0 {
            continue;
        }

        let stat = log_rank_stat(time, status, &left_idx, &right_idx);
        if stat > best_stat {
            best_stat = stat;
            best_feature = Some(f);
            best_value = Some(median);
        }
    }

    if best_stat == 0.0 {
        return node; // cannot split
    }

    node.split_feature = best_feature;
    node.split_value = best_value;

    // Split recursively
    let left_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| data[[i,best_feature.unwrap()]] <= best_value.unwrap())
        .collect();
    let right_idx: Vec<usize> = indices.iter()
        .cloned()
        .filter(|&i| data[[i,best_feature.unwrap()]] > best_value.unwrap())
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

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut trees = Vec::new();
    let mut feature_importance: HashMap<usize, f64> = HashMap::new();

    for f in 0..n_features {
        feature_importance.insert(f, 0.0);
    }

    for _ in 0..config.n_trees {
        // bootstrap sample
        let indices: Vec<usize> = (0..data.nrows()).collect();
        let sample_indices: Vec<usize> = indices.choose_multiple(&mut rng, data.nrows()).cloned().collect();

        let tree = build_tree(data, time, status, sample_indices, config.min_node_size, max_features, &mut rng);
        trees.push(tree);

        // Simple variable importance: count how many times each feature was used
        for t in &trees {
            increment_importance(&t, &mut feature_importance);
        }
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

