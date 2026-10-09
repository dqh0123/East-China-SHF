import math
import pandas as pd
import numpy as np
from typing import List, Dict


class QuadtreeNode:
    def __init__(self, xmin, ymin, xmax, ymax, depth=0, cell_id=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.depth = depth
        self.cell_id = cell_id
        self.children = []

    def bbox(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def center(self):
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def split(self):
        cx, cy = self.center()
        d = self.depth + 1
        return [
            QuadtreeNode(self.xmin, self.ymin, cx, cy, d),  # SW
            QuadtreeNode(cx, self.ymin, self.xmax, cy, d),  # SE
            QuadtreeNode(self.xmin, cy, cx, self.ymax, d),  # NW
            QuadtreeNode(cx, cy, self.xmax, self.ymax, d),  # NE
        ]


def points_in_bbox(df_points: pd.DataFrame, bbox) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bbox
    return df_points[(df_points['Lon'] >= xmin) & (df_points['Lon'] <= xmax)
                     & (df_points['Lat'] >= ymin) & (df_points['Lat'] <= ymax)]


def geo_stats_in_bbox(df_geo: pd.DataFrame, bbox, geo_cols: List[str]) -> Dict[str, float]:
    subset = points_in_bbox(df_geo, bbox)
    if subset.empty:
        return {'n_cells': 0, 'geo_var': 0.0}
    # compute mean variance across geo columns (ignore NaNs)
    var_per_col = subset[geo_cols].var(ddof=0).replace({np.nan: 0.0})
    mean_var = var_per_col.mean()
    return {'n_cells': len(subset), 'geo_var': float(mean_var)}


def build_quadtree(df_shf: pd.DataFrame,
                   df_geo: pd.DataFrame,
                   max_depth=6,
                   min_size=0.5,
                   point_thresh=10,
                   var_factor=0.5,
                   geo_cols: List[str] = None,
                   bounds: tuple = None) -> List[QuadtreeNode]:
    # Rename columns if necessary
    if 'Lon' not in df_shf.columns and 'Longitude' in df_shf.columns:
        df_shf = df_shf.rename(columns={'Longitude': 'Lon', 'Latitude': 'Lat'})
    if 'Lon' not in df_geo.columns and 'Longitude' in df_geo.columns:
        df_geo = df_geo.rename(columns={'Longitude': 'Lon', 'Latitude': 'Lat'})

    # determine geo columns used for variance:
    #   - if geo_cols is None -> all non Lon/Lat columns (original behaviour)
    #   - else -> only the user-specified target columns (that actually exist)
    if geo_cols is None:
        geo_cols = [c for c in df_geo.columns if c not in ['Lat', 'Lon']]
    else:
        geo_cols = [c for c in geo_cols if c in df_geo.columns]
    if not geo_cols:
        raise ValueError('geo_cols is empty after filtering against df_geo columns')

    # spatial bounds: manual bounds take priority, otherwise derive from data
    if bounds is not None:
        lon_min, lat_min, lon_max, lat_max = bounds
    else:
        lon_min = min(df_shf['Lon'].min(), df_geo['Lon'].min())
        lon_max = max(df_shf['Lon'].max(), df_geo['Lon'].max())
        lat_min = min(df_shf['Lat'].min(), df_geo['Lat'].min())
        lat_max = max(df_shf['Lat'].max(), df_geo['Lat'].max())

    root = QuadtreeNode(lon_min, lat_min, lon_max, lat_max, depth=0, cell_id='0')

    # baseline global geo variance (computed on the selected columns)
    global_geo_var = df_geo[geo_cols].var(ddof=0).replace({np.nan: 0.0}).mean()
    var_thresh = float(global_geo_var * var_factor)

    leaves = []
    stack = [root]
    next_id = 1

    while stack:
        node = stack.pop()
        bbox = node.bbox()
        # count SHF points inside
        pts = points_in_bbox(df_shf, bbox)
        n_points = len(pts)
        # geo stats
        gstats = geo_stats_in_bbox(df_geo, bbox, geo_cols)
        geo_var = gstats['geo_var']

        # stop conditions
        w = node.width()
        h = node.height()
        min_dimension = min(w, h)
        should_split = False
        if node.depth < max_depth and min_dimension > min_size:
            if n_points > point_thresh:
                should_split = True
            elif geo_var > var_thresh:
                should_split = True

        if should_split:
            children = node.split()
            for ch in children:
                ch.cell_id = f"{next_id}"
                next_id += 1
                stack.append(ch)
        else:
            # leaf
            node_info = {
                'cell_id': node.cell_id,
                'xmin': node.xmin,
                'ymin': node.ymin,
                'xmax': node.xmax,
                'ymax': node.ymax,
                'center_x': node.center()[0],
                'center_y': node.center()[1],
                'width': node.width(),
                'height': node.height(),
                'depth': node.depth,
                'n_points': n_points,
                'geo_var': geo_var,
                'n_geo_cells': gstats['n_cells']
            }
            leaves.append(node_info)

    return leaves


def main():
    # load datasets (paths matched to the notebook)
    df_SHF = pd.read_excel('..\\data\\SHF.xls')
    df_GEO = pd.read_excel('..\\output\\merged_all_datasets.xlsx')

    # ensure columns - rename if necessary
    if 'Lon' not in df_SHF.columns:
        if 'Longitude' in df_SHF.columns:
            df_SHF = df_SHF.rename(columns={'Longitude': 'Lon', 'Latitude': 'Lat'})
        else:
            raise ValueError('df_SHF must have Lon/Latitude or Longitude/Latitude columns')
    
    if 'Lon' not in df_GEO.columns:
        if 'Longitude' in df_GEO.columns:
            df_GEO = df_GEO.rename(columns={'Longitude': 'Lon', 'Latitude': 'Lat'})
        else:
            raise ValueError('df_GEO must have Lon/Latitude or Longitude/Latitude columns')

    # build quadtree
    leaves = build_quadtree(df_SHF, df_GEO,
                            max_depth=8,
                            min_size=0.5,
                            point_thresh=10,
                            var_factor=0.5)

    mesh_df = pd.DataFrame(leaves)
    # sort by depth then id
    mesh_df = mesh_df.sort_values(['depth', 'cell_id']).reset_index(drop=True)

    # save
    out_path = '..\\output\\quadtree_mesh.csv'
    mesh_df.to_csv(out_path, index=False)
    print(f"Saved quadtree mesh to: {out_path}")

    return mesh_df


if __name__ == '__main__':
    main()
