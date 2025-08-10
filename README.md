# Graph Neural Networks — Playlist Graph

Learning GNNs using **Spotify playlists** as the dataset. I first planned a Goodreads book graph but dropped it due to ToS. Playlists are a **less obvious** choice than citation or social graphs, but they work well for understanding the end-to-end pipeline on data I control.

## Graph definition
- **Nodes:** playlists  
- **Edges:** connect playlists that share ≥1 song (optionally weighted by the number of shared tracks)

## Goals
- Build the playlist–playlist graph 
- Train simple GNN baselines (GCN/GraphSAGE)  
- Include small test scripts while I get back up to speed with PyTorch and graph GNNs
- predict links on playlist graph.


## Status
Work in progress; expect changes.


