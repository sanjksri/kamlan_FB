import io
import os
import tempfile
from datetime import datetime
from typing import Any, List, Tuple

import streamlit as st

# Optional heavy deps; import lazily/defensively
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
except Exception:  # pragma: no cover
    gpd = None
    Polygon = None

try:
    import leafmap.foliumap as leaf_folium
except Exception:  # pragma: no cover
    leaf_folium = None

try:
    import geemap.foliumap as gee_folium  # noqa: F401  (not strictly required in fallbacks)
except Exception:  # pragma: no cover
    gee_folium = None

try:
    import ee  # noqa: F401
except Exception:  # pragma: no cover
    ee = None


def force_stop() -> None:
    st.stop()


def _make_dummy_gdf() -> "gpd.GeoDataFrame":
    if gpd is None or Polygon is None:
        raise RuntimeError("geopandas/shapely unavailable to construct geometry.")

    # Simple square near (0, 0)
    poly = Polygon([(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)])
    gdf = gpd.GeoDataFrame({"Name": ["AOI" ]}, geometry=[poly], crs="EPSG:4326")
    return gdf


def get_gdf_from_file_url(file_or_url: Any) -> "gpd.GeoDataFrame":
    """Load GeoJSON/KML from URL or UploadedFile; fallback to a dummy polygon."""
    if gpd is None:
        return _make_dummy_gdf()

    # Streamlit UploadedFile has .read() and .name
    try:
        # String URL
        if isinstance(file_or_url, str):
            try:
                return gpd.read_file(file_or_url)
            except Exception:
                # Fallback dummy
                return _make_dummy_gdf()

        # Uploaded file-like
        if hasattr(file_or_url, "read"):
            suffix = ".geojson"
            if hasattr(file_or_url, "name") and str(file_or_url.name).lower().endswith(".kml"):
                suffix = ".kml"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_or_url.read())
                tmp_path = tmp.name
            try:
                return gpd.read_file(tmp_path)
            except Exception:
                return _make_dummy_gdf()
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    except Exception:
        return _make_dummy_gdf()


def preprocess_gdf(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gpd is None:
        return _make_dummy_gdf()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    # Drop empties
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def is_valid_polygon(gdf: "gpd.GeoDataFrame") -> bool:
    if gpd is None:
        return True
    geom = gdf.geometry.iloc[0]
    return geom.geom_type in {"Polygon", "MultiPolygon"}


def to_best_crs(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gpd is None:
        return _make_dummy_gdf()
    # Use Web Mercator (meters) as a simple universal CRS for area/length
    try:
        return gdf.to_crs(3857)
    except Exception:
        return gdf


def write_info(html: str, center_align: bool = False) -> None:
    if center_align:
        st.markdown(f"<div style='text-align:center'>{html}</div>", unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def add_geometry_to_maps(maps: List[Any], geometry_gdf: "gpd.GeoDataFrame", buffer_geometry_gdf: "gpd.GeoDataFrame", opacity: float = 0.3) -> None:
    for m in maps:
        try:
            if hasattr(m, "add_gdf"):
                m.add_gdf(geometry_gdf, layer_name="Geometry", style_function=lambda x: {"color": "blue", "fillOpacity": opacity})
                if buffer_geometry_gdf is not None:
                    m.add_gdf(buffer_geometry_gdf, layer_name="Buffer", style_function=lambda x: {"color": "red", "fillOpacity": opacity/2})
        except Exception:
            continue


def one_time_setup() -> None:
    # Try to initialize Earth Engine if available; ignore failures in Cloud Run
    try:
        if ee is not None:
            ee.Initialize()
    except Exception:
        pass


def get_dem_slope_maps(ee_geometry: Any, wayback_url: str, wayback_title: str) -> Tuple[Any, Any]:
    """Return simple Leafmap maps as placeholders (no EE dependency)."""
    if leaf_folium is None:
        return None, None
    m1 = leaf_folium.Map()
    m2 = leaf_folium.Map()
    try:
        m1.add_tile_layer(wayback_url, name=wayback_title, attribution="Esri")
        m2.add_tile_layer(wayback_url, name=wayback_title, attribution="Esri")
    except Exception:
        pass
    return m1, m2


# The following are placeholders so the app doesn’t crash if users press buttons.
def process_date(*args, **kwargs) -> None:  # pragma: no cover
    # No-op placeholder. Real implementation would compute indices with EE.
    return None


def get_histogram(*args, **kwargs):  # pragma: no cover
    return [0, 0, 0, 0, 0, 0, 0], []


def daterange_str_to_year(daterange: Any) -> str:
    try:
        # If daterange is a tuple of pandas timestamps
        start = getattr(daterange, "left", None) or (daterange[0] if isinstance(daterange, (list, tuple)) else None)
        if start is not None:
            return str(getattr(start, "year", datetime.now().year))
    except Exception:
        pass
    return str(datetime.now().year)


def daterange_str_to_dates(daterange: Any) -> Tuple[datetime, datetime]:  # pragma: no cover
    try:
        start = getattr(daterange, "left", None) or (daterange[0] if isinstance(daterange, (list, tuple)) else None)
        end = getattr(daterange, "right", None) or (daterange[1] if isinstance(daterange, (list, tuple)) else None)
        if start is None or end is None:
            now = datetime.utcnow()
            return now, now
        return start.to_pydatetime(), end.to_pydatetime()
    except Exception:
        now = datetime.utcnow()
        return now, now


def show_visitor_counter(path: str) -> None:  # pragma: no cover
    # Lightweight placeholder
    return None


def show_credits() -> None:  # pragma: no cover
    st.caption("Deployed on Cloud Run • Served via Firebase Hosting")


__all__ = [
    "force_stop",
    "get_gdf_from_file_url",
    "preprocess_gdf",
    "is_valid_polygon",
    "to_best_crs",
    "write_info",
    "add_geometry_to_maps",
    "one_time_setup",
    "get_dem_slope_maps",
    "process_date",
    "get_histogram",
    "daterange_str_to_year",
    "daterange_str_to_dates",
    "show_visitor_counter",
    "show_credits",
]


