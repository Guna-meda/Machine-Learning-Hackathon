"""Folium map of one trip: GPS points, road network around the track, reference route, optional prediction."""
import numpy as np
import folium


def _to_lonlat_list(xy_lonlat):
    return [[float(lat), float(lon)] for lon, lat in xy_lonlat]


def trip_map(trip, ref_arcs=None, pred_arcs=None, max_network_arcs=3000, title=None):
    """trip: Trip object.  ref_arcs / pred_arcs: ordered arc id arrays (or None)."""
    track = trip.track
    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    # fit to the GPS extent
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])

    # road network near the track: arcs whose midpoint is within NEAR_M of a GPS point (keeps the HTML small and relevant)
    from scipy.spatial import cKDTree
    mid = (trip.nodes_xy[trip.arcs[:, 0]] + trip.nodes_xy[trip.arcs[:, 1]]) / 2
    dist, _ = cKDTree(trip.track_xy[::3]).query(mid)
    near = np.where(dist <= 120.0)[0]
    if len(near) > max_network_arcs:
        near = near[np.argsort(dist[near])[:max_network_arcs]]
    a = trip.nodes[trip.arcs[:, 0]]; b = trip.nodes[trip.arcs[:, 1]]
    net = folium.FeatureGroup(name=f"Road network near the track ({len(near)} arcs)", show=True)
    for i in near:
        folium.PolyLine([[a[i, 1], a[i, 0]], [b[i, 1], b[i, 0]]], color="#9aa0a6", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    def route_layer(arc_ids, name, color, weight, dash=None):
        fg = folium.FeatureGroup(name=name, show=True)
        pts = []
        for aid in arc_ids:
            u, v = trip.nodes[trip.arcs[aid, 0]], trip.nodes[trip.arcs[aid, 1]]
            folium.PolyLine([[u[1], u[0]], [v[1], v[0]]], color=color, weight=weight, opacity=0.9, dash_array=dash).add_to(fg)
        fg.add_to(m)

    if ref_arcs is not None:
        route_layer(ref_arcs, "Reference route", "#1a9850", 6)
    if pred_arcs is not None:
        route_layer(pred_arcs, "Predicted route", "#d73027", 3, dash="6 6")

    gps = folium.FeatureGroup(name="GPS points", show=True)
    step = max(1, len(track) // 800)       # thin very long tracks
    for lat, lon, t in track.iloc[::step][["lat", "lon", "t"]].itertuples(index=False):
        folium.CircleMarker([lat, lon], radius=2, color="#2166ac", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([track.lat.iloc[0], track.lon.iloc[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([track.lat.iloc[-1], track.lon.iloc[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = f"""<div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white; padding: 8px 12px;
      border: 1px solid #888; font-size: 13px;"><b>{title or 'Trip ' + trip.trip_id}</b><br>
      <span style="color:#2166ac">&#9679;</span> GPS points<br>
      <span style="color:#9aa0a6">&#9473;</span> Road network<br>
      {'<span style="color:#1a9850">&#9473;</span> Reference route<br>' if ref_arcs is not None else ''}
      {'<span style="color:#d73027">&#9473;</span> Predicted route<br>' if pred_arcs is not None else ''}</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
