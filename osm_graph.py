from pyproj import Geod
import geopandas as gpd
import osmnx as ox
import networkx as nx
import shapely.geometry as geom

class OSMGraph:
    #sim = GTFS_OSM_DES(gtfs_folder=GTFS_FOLDER, graph_path="map.xml")
    def __init__(self, graph_xml: str):
        self.G = ox.graph_from_xml(graph_xml)
        self.nodes, _ = ox.graph_to_gdfs(self.G)
        # cache for shortest path distances (node_u, node_v) -> meters
        self._sp_cache = {}
        # ensure every edge has a metric 'length' attribute (meters)
        self._ensure_edge_lengths()

    def _ensure_edge_lengths(self):
        """Ensure all edges in the graph have a 'length' attribute in meters.

        If 'length' exists and seems numeric, keep it; otherwise compute from
        edge geometry or from node coordinates using Geod.
        """
        geod = Geod(ellps="WGS84")
        for u, v in self.G.edges():
            ed = self.G.get_edge_data(u, v)
            if ed is None:
                continue
            # In a MultiGraph get_edge_data returns dict of key->data; in Graph it returns data dict
            if isinstance(ed, dict) and any(isinstance(k, (int, str)) for k in ed.keys()):
                # likely a MultiGraph: iterate over key,data pairs
                items = ed.items()
            else:
                # single edge: use a single artificial key
                items = [(None, ed)]
            for key, data in items:
             # if 'length' already present and > 0, assume it's in meters
             length_val = data.get("length")
             if length_val is not None:
                 try:
                     if float(length_val) > 0:
                         data["length"] = float(length_val)
                         continue
                 except Exception:
                     pass

             # try geometry
             geom_attr = data.get("geometry")
             if geom_attr is not None:
                 try:
                     coords = list(geom_attr.coords)
                     seg_m = 0.0
                     for i in range(len(coords) - 1):
                         lon1, lat1 = coords[i]
                         lon2, lat2 = coords[i + 1]
                         _, _, d = geod.inv(lon1, lat1, lon2, lat2)
                         seg_m += abs(d)
                     data["length"] = seg_m
                     continue
                 except Exception:
                     pass

             # fallback: use node coordinates
             nu = self.G.nodes.get(u, {})
             nv = self.G.nodes.get(v, {})
             try:
                 lon1 = nu.get("x") if nu.get("x") is not None else nu.get("lon")
                 lat1 = nu.get("y") if nu.get("y") is not None else nu.get("lat")
                 lon2 = nv.get("x") if nv.get("x") is not None else nv.get("lon")
                 lat2 = nv.get("y") if nv.get("y") is not None else nv.get("lat")
                 if None not in (lon1, lat1, lon2, lat2):
                     _, _, d = geod.inv(lon1, lat1, lon2, lat2)
                     data["length"] = abs(d)
                     continue
             except Exception:
                 pass

             # As a last resort, set to None
             data["length"] = None

    def snap_stops(self, stops_df):
        gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=gpd.points_from_xy(
                stops_df.stop_lon.astype(float),
                stops_df.stop_lat.astype(float)
            ),
            crs="EPSG:4326"
        )
        nodes = ox.nearest_nodes(self.G, gdf.geometry.x, gdf.geometry.y)
        stops_df = stops_df.copy()
        stops_df["node_id"] = nodes
        return stops_df

    def shortest_path_distance(self, node_u, node_v):
        """Return shortest path length in meters between two graph node IDs using cached results."""
        if node_u is None or node_v is None:
            return None
        key = (int(node_u), int(node_v))
        if key in self._sp_cache:
            return self._sp_cache[key]
        try:
            d = nx.shortest_path_length(self.G, node_u, node_v, weight="length")
            if d is None:
                self._sp_cache[key] = None
                return None
            self._sp_cache[key] = float(d)
            return float(d)
        except Exception:
            self._sp_cache[key] = None
            return None