import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import networkx as nx
from itertools import combinations
import os
import matplotlib.pyplot as plt
import re
# The datetime import is no longer needed

def strip_emojis(text):
    """
    Strips emojis and other non-text symbols from a string.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).strip()

def build_playlist_similarity_graph(sp, playlist_limit=20, country='GB'):
    """
    Builds a graph using featured playlists as nodes, connected by shared songs.
    """
    G = nx.Graph()
    playlist_to_tracks = {}

    print(f"Finding featured playlists for country '{country}'...")
    try:

        results = sp.featured_playlists(country=country, limit=playlist_limit)
        playlists = results['playlists']['items']
    except Exception as e:
        print(f"Could not fetch featured playlists. Error: {e}")
        return G


    for playlist in playlists:
        if playlist:
            playlist_id = playlist['id']
            playlist_name = strip_emojis(playlist['name'])
            G.add_node(playlist_id, name=playlist_name)
            print(f"  Fetching tracks from: {playlist_name}")
            try:
                track_items = sp.playlist_tracks(playlist_id, market=country)['items']
                track_ids = {item['track']['id'] for item in track_items if item.get('track') and item['track'].get('id')}
                playlist_to_tracks[playlist_id] = track_ids
            except Exception as e:
                print(f"    Could not fetch tracks for playlist {playlist_name}. Error: {e}")

    song_to_playlists = {}
    for playlist_id, track_ids in playlist_to_tracks.items():
        for track_id in track_ids:
            if track_id not in song_to_playlists:
                song_to_playlists[track_id] = []
            song_to_playlists[track_id].append(track_id)

    print("\nBuilding graph from shared songs...")
    for song, shared_playlists in song_to_playlists.items():
        if len(shared_playlists) > 1:
            for p1, p2 in combinations(shared_playlists, 2):
                if G.has_edge(p1, p2):
                    G[p1][p2]['weight'] += 1
                else:
                    G.add_edge(p1, p2, weight=1)

    return G

# --- Run the script ---
if __name__ == "__main__":
    token = os.environ.get('SPOTIFY_API_TOKEN_FOR_GRAPHS')
    if not token:
        print("Error: SPOTIFY_API_TOKEN_FOR_GRAPHS environment variable not set.")
    else:
        try:
            sp = spotipy.Spotify(auth=token)

            playlist_graph = build_playlist_similarity_graph(sp, playlist_limit=50, country='GB')

            print("\nGraph building complete!")
            print(f"  - Nodes (Playlists): {playlist_graph.number_of_nodes()}")
            print(f"  - Edges (Shared Songs): {playlist_graph.number_of_edges()}")

            if playlist_graph.number_of_nodes() > 0:
                plt.figure(figsize=(15, 15))
                pos = nx.spring_layout(playlist_graph, k=0.5, iterations=50)
                labels = nx.get_node_attributes(playlist_graph, 'name')
                weights = [playlist_graph[u][v]['weight'] for u, v in playlist_graph.edges()]

                nx.draw(playlist_graph, pos, labels=labels, with_labels=True, width=weights, edge_color='lightblue', node_color='skyblue', font_size=8)
                plt.title("Playlist Similarity Graph (Featured Playlists)")
                plt.show()

        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 401:
                print("\nAuthentication failed. Your token has likely expired.")
            else:
                print(f"\nAn error occurred: {e}")
