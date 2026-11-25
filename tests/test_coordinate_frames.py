"""WGS84 / CGCS2000 / LocalFrame 转换精度与兼容别名。"""
from __future__ import annotations

import inspect
import math
import tomllib
from pathlib import Path

import pytest
from pyproj import Geod

from common.exception import CoordinateFrameException
from envs.state.spatial import (
    CGCS2000_A,
    CGCS2000_RF,
    CGCS2000Frame,
    GeoPoint,
    LocalFrame,
    WGS84_A,
    WGS84_RF,
    WGS84Frame,
    XYZPosition,
)


WGS84_GEOD = Geod(ellps="WGS84")
CGCS2000_GEOD = Geod(a=6378137.0, rf=298.257222101)
ROOT = Path(__file__).resolve().parents[1]

_SAMPLES = (
    (118.7969, 32.0603, 20.0),
    (0.0, 0.0, 0.0),
    (10.0, 70.0, 100.0),
    (20.0, 89.5, 50.0),
)


def _horizontal_m(geod: Geod, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _az12, _az21, dist = geod.inv(lon1, lat1, lon2, lat2)
    return abs(dist)


def _offset_point(geod: Geod, lon: float, lat: float, azimuth_deg: float, distance_m: float):
    dest_lon, dest_lat, _back = geod.fwd(lon, lat, azimuth_deg, distance_m)
    if dest_lon > 180.0:
        dest_lon -= 360.0
    elif dest_lon <= -180.0:
        dest_lon += 360.0
    return dest_lon, dest_lat


def test_ellipsoid_constants_are_defined_once() -> None:
    assert WGS84Frame._a is WGS84_A
    assert CGCS2000Frame._a is CGCS2000_A
    assert WGS84_A == 6378137.0
    assert CGCS2000_A == 6378137.0
    assert WGS84_RF == 298.257223563
    assert CGCS2000_RF == 298.257222101
    assert WGS84Frame._a == CGCS2000Frame._a
    assert WGS84_RF != CGCS2000_RF


def test_geo_point_aliases_and_legacy_names() -> None:
    assert GeoPoint is WGS84Frame
    assert XYZPosition is LocalFrame
    assert GeoPoint is not CGCS2000Frame
    point = GeoPoint(lon=118.8, lat=32.1, alt=12.0)
    assert point.longitude == pytest.approx(118.8)
    assert point.latitude == pytest.approx(32.1)
    assert point.ellipsoidal_height == pytest.approx(12.0)
    payload = point.to_dict()
    assert "lon" not in payload
    assert "lat" not in payload
    assert "alt" not in payload
    assert payload["longitude"] == pytest.approx(118.8)
    assert payload["latitude"] == pytest.approx(32.1)
    assert payload["ellipsoidal_height"] == pytest.approx(12.0)
    restored = WGS84Frame.from_dict({"lon": 1.0, "lat": 2.0, "alt": 3.0})
    assert restored.longitude == pytest.approx(1.0)
    assert restored.latitude == pytest.approx(2.0)
    assert restored.ellipsoidal_height == pytest.approx(3.0)


def test_geo_point_legacy_operators_keep_historical_formula() -> None:
    p1 = GeoPoint(lon=118.0, lat=32.0, alt=10.0)
    p2 = GeoPoint(lon=118.1, lat=32.1, alt=20.0)
    p3 = GeoPoint(lon=118.2, lat=31.9, alt=0.0)
    earthe = 0.0818191908426
    radius_earth = 6378140.0
    omge = 0.99330562000987
    pi = 3.14159265358979
    deg2rad = pi / 180.0
    lat = p1.lat * deg2rad
    lon = p1.lon * deg2rad
    d = earthe * math.sin(lat)
    n = radius_earth / math.sqrt(1 - math.pow(d, 2))
    nph = n + p1.alt
    expected = XYZPosition(
        nph * math.cos(lat) * math.cos(lon),
        nph * math.cos(lat) * math.sin(lon),
        (omge * n + p1.alt) * math.sin(lat),
    )
    xyz = p1.to_xyz()
    assert xyz.x == pytest.approx(expected.x)
    assert xyz.y == pytest.approx(expected.y)
    assert xyz.z == pytest.approx(expected.z)
    assert GeoPoint.distance(p1, p2) == pytest.approx(XYZPosition.distance(p1.to_xyz(), p2.to_xyz()))
    assert GeoPoint.central_angle(p1, p2) > 0.0
    assert GeoPoint.angle(p1, p2, p3) >= 0.0


def test_earth_to_earth_signatures_have_no_origin() -> None:
    wgs_params = inspect.signature(WGS84Frame.to_cgcs2000).parameters
    cgcs_params = inspect.signature(CGCS2000Frame.to_wgs84).parameters
    assert "origin" not in wgs_params
    assert "origin" not in cgcs_params
    assert "x_axis_bearing_deg" not in wgs_params
    assert "x_axis_bearing_deg" not in cgcs_params
    local_params = inspect.signature(WGS84Frame.to_local).parameters
    assert "origin" in local_params
    assert local_params["origin"].kind is inspect.Parameter.KEYWORD_ONLY
    assert local_params["x_axis_bearing_deg"].default == 90.0


@pytest.mark.parametrize("lon,lat,height", _SAMPLES)
def test_wgs84_cgcs2000_roundtrip(lon: float, lat: float, height: float) -> None:
    wgs = WGS84Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)
    cgcs = wgs.to_cgcs2000()
    back = cgcs.to_wgs84()
    assert _horizontal_m(WGS84_GEOD, wgs.longitude, wgs.latitude, back.longitude, back.latitude) < 2e-3
    assert abs(back.ellipsoidal_height - wgs.ellipsoidal_height) < 1e-3

    cgcs_src = CGCS2000Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)
    wgs_mid = cgcs_src.to_wgs84()
    cgcs_back = wgs_mid.to_cgcs2000()
    assert _horizontal_m(
        CGCS2000_GEOD,
        cgcs_src.longitude,
        cgcs_src.latitude,
        cgcs_back.longitude,
        cgcs_back.latitude,
    ) < 2e-3
    assert abs(cgcs_back.ellipsoidal_height - cgcs_src.ellipsoidal_height) < 1e-3


def test_wgs84_to_cgcs2000_keeps_ecef_not_copied_geodetic() -> None:
    wgs = WGS84Frame(longitude=118.7969, latitude=32.0603, ellipsoidal_height=20.0)
    cgcs = wgs.to_cgcs2000()
    wx, wy, wz = wgs._to_ecef()
    cx, cy, cz = cgcs._to_ecef()
    assert math.dist((float(wx), float(wy), float(wz)), (float(cx), float(cy), float(cz))) < 1e-3


@pytest.mark.parametrize("lon,lat,height", _SAMPLES)
@pytest.mark.parametrize("distance_m,limit_m", ((300.0, 1e-2), (50_000.0, 1.0)))
def test_wgs84_local_roundtrip(
        lon: float,
        lat: float,
        height: float,
        distance_m: float,
        limit_m: float,
) -> None:
    origin = WGS84Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)
    dest_lon, dest_lat = _offset_point(WGS84_GEOD, lon, lat, 45.0, distance_m)
    point = WGS84Frame(longitude=dest_lon, latitude=dest_lat, ellipsoidal_height=height + 8.0)
    local = point.to_local(origin=origin)
    restored = local.to_wgs84(origin=origin)
    assert _horizontal_m(
        WGS84_GEOD,
        point.longitude,
        point.latitude,
        restored.longitude,
        restored.latitude,
    ) < limit_m
    assert abs(restored.ellipsoidal_height - point.ellipsoidal_height) < limit_m
    back_local = restored.to_local(origin=origin)
    assert math.dist((local.x, local.y, local.z), (back_local.x, back_local.y, back_local.z)) < limit_m


@pytest.mark.parametrize("lon,lat,height", _SAMPLES)
@pytest.mark.parametrize("distance_m,limit_m", ((300.0, 1e-2), (50_000.0, 1.0)))
def test_cgcs2000_local_roundtrip(
        lon: float,
        lat: float,
        height: float,
        distance_m: float,
        limit_m: float,
) -> None:
    origin = CGCS2000Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)
    dest_lon, dest_lat = _offset_point(CGCS2000_GEOD, lon, lat, 30.0, distance_m)
    point = CGCS2000Frame(longitude=dest_lon, latitude=dest_lat, ellipsoidal_height=height + 12.0)
    local = point.to_local(origin=origin)
    restored = local.to_cgcs2000(origin=origin)
    assert _horizontal_m(
        CGCS2000_GEOD,
        point.longitude,
        point.latitude,
        restored.longitude,
        restored.latitude,
    ) < limit_m
    assert abs(restored.ellipsoidal_height - point.ellipsoidal_height) < limit_m


def test_default_enu_100m_east_and_north() -> None:
    origin = WGS84Frame(longitude=118.7969, latitude=32.0603, ellipsoidal_height=20.0)
    east_lon, east_lat = _offset_point(WGS84_GEOD, origin.longitude, origin.latitude, 90.0, 100.0)
    north_lon, north_lat = _offset_point(WGS84_GEOD, origin.longitude, origin.latitude, 0.0, 100.0)
    east = WGS84Frame(
        longitude=east_lon,
        latitude=east_lat,
        ellipsoidal_height=origin.ellipsoidal_height,
    ).to_local(origin=origin)
    north = WGS84Frame(
        longitude=north_lon,
        latitude=north_lat,
        ellipsoidal_height=origin.ellipsoidal_height,
    ).to_local(origin=origin)
    assert east.x > 0.0
    assert abs(east.x - 100.0) < 1e-2
    assert abs(east.y) < 1e-2
    assert north.y > 0.0
    assert abs(north.y - 100.0) < 1e-2
    assert abs(north.x) < 1e-2


def test_origin_maps_to_zero_and_height_to_up() -> None:
    wgs_origin = WGS84Frame(longitude=118.7969, latitude=32.0603, ellipsoidal_height=20.0)
    wgs_zero = wgs_origin.to_local(origin=wgs_origin)
    assert abs(wgs_zero.x) < 1e-9
    assert abs(wgs_zero.y) < 1e-9
    assert abs(wgs_zero.z) < 1e-9
    raised = WGS84Frame(
        longitude=wgs_origin.longitude,
        latitude=wgs_origin.latitude,
        ellipsoidal_height=120.0,
    ).to_local(origin=wgs_origin)
    assert abs(raised.x) < 1e-2
    assert abs(raised.y) < 1e-2
    assert raised.z == pytest.approx(100.0, abs=1e-2)

    cgcs_origin = CGCS2000Frame(longitude=118.7969, latitude=32.0603, ellipsoidal_height=20.0)
    cgcs_zero = cgcs_origin.to_local(origin=cgcs_origin)
    assert abs(cgcs_zero.x) < 1e-9
    assert abs(cgcs_zero.y) < 1e-9
    assert abs(cgcs_zero.z) < 1e-9


def test_bearing_zero_x_north_y_west() -> None:
    origin = WGS84Frame(longitude=118.7969, latitude=32.0603, ellipsoidal_height=0.0)
    north_lon, north_lat = _offset_point(WGS84_GEOD, origin.longitude, origin.latitude, 0.0, 100.0)
    east_lon, east_lat = _offset_point(WGS84_GEOD, origin.longitude, origin.latitude, 90.0, 100.0)
    north = WGS84Frame(longitude=north_lon, latitude=north_lat, ellipsoidal_height=0.0).to_local(
        origin=origin,
        x_axis_bearing_deg=0.0,
    )
    east = WGS84Frame(longitude=east_lon, latitude=east_lat, ellipsoidal_height=0.0).to_local(
        origin=origin,
        x_axis_bearing_deg=0.0,
    )
    assert north.x == pytest.approx(100.0, abs=1e-2)
    assert abs(north.y) < 1e-2
    assert abs(east.x) < 1e-2
    assert east.y == pytest.approx(-100.0, abs=1e-2)


def test_mismatched_origin_types_fail() -> None:
    wgs = WGS84Frame(longitude=118.0, latitude=32.0, ellipsoidal_height=0.0)
    cgcs = CGCS2000Frame(longitude=118.0, latitude=32.0, ellipsoidal_height=0.0)
    local = LocalFrame(x=10.0, y=0.0, z=0.0)
    with pytest.raises(CoordinateFrameException):
        wgs.to_local(origin=cgcs)
    with pytest.raises(CoordinateFrameException):
        cgcs.to_local(origin=wgs)
    with pytest.raises(CoordinateFrameException):
        local.to_wgs84(origin=cgcs)
    with pytest.raises(CoordinateFrameException):
        local.to_cgcs2000(origin=wgs)
    with pytest.raises(CoordinateFrameException):
        wgs.to_local(origin=local)


def test_range_and_nonfinite_values_fail() -> None:
    with pytest.raises(ValueError):
        WGS84Frame(longitude=181.0, latitude=0.0, ellipsoidal_height=0.0)
    with pytest.raises(ValueError):
        WGS84Frame(longitude=0.0, latitude=91.0, ellipsoidal_height=0.0)
    WGS84Frame(longitude=180.0, latitude=90.0, ellipsoidal_height=0.0)
    WGS84Frame(longitude=-180.0, latitude=-90.0, ellipsoidal_height=0.0)
    with pytest.raises(ValueError):
        WGS84Frame(longitude=math.nan, latitude=0.0, ellipsoidal_height=0.0)
    with pytest.raises(ValueError):
        LocalFrame(x=math.inf, y=0.0, z=0.0)
    origin = WGS84Frame(longitude=0.0, latitude=0.0, ellipsoidal_height=0.0)
    with pytest.raises(ValueError):
        origin.to_local(origin=origin, x_axis_bearing_deg=math.nan)


def test_pole_reverse_stays_bounded() -> None:
    pole = WGS84Frame(longitude=0.0, latitude=90.0, ellipsoidal_height=10.0)
    cgcs = pole.to_cgcs2000()
    assert abs(cgcs.latitude) <= 90.0
    local = pole.to_local(origin=pole)
    assert math.isfinite(local.x)
    assert math.isfinite(local.y)
    assert math.isfinite(local.z)
    south_lon, south_lat = _offset_point(WGS84_GEOD, 0.0, 89.9, 180.0, 300.0)
    near = WGS84Frame(longitude=south_lon, latitude=south_lat, ellipsoidal_height=10.0)
    restored = near.to_local(origin=pole).to_wgs84(origin=pole)
    assert abs(restored.latitude) <= 90.0


def test_sdk_runtime_has_no_pyproj_dependency() -> None:
    sdk_project = ROOT / "sdk" / "pyproject.toml"
    with sdk_project.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    assert all("pyproj" not in item for item in dependencies)
    import envs.state.spatial as spatial
    assert not hasattr(spatial, "WGS84LocalFrame")
