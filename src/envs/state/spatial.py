"""空间坐标：WGS84 / CGCS2000 / 局部 Frame 与兼容别名。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from dataclasses_json import DataClassJsonMixin

from common.exception import CoordinateFrameException

# 椭球常量只在本模块定义一份。
WGS84_A = 6378137.0
WGS84_RF = 298.257223563
WGS84_F = 1.0 / WGS84_RF
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

CGCS2000_A = 6378137.0
CGCS2000_RF = 298.257222101
CGCS2000_F = 1.0 / CGCS2000_RF
CGCS2000_E2 = CGCS2000_F * (2.0 - CGCS2000_F)

_MAX_ECEF_ITERS = 12
_LAT_CONV_RAD = 1e-12


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _validate_lon_lat(longitude: float, latitude: float) -> None:
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {latitude}")


def _wrap_longitude(longitude: float) -> float:
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def _geodetic_to_ecef(
        longitude_deg: float,
        latitude_deg: float,
        height: float,
        a: float,
        e2: float,
) -> tuple[np.float64, np.float64, np.float64]:
    lon = np.deg2rad(np.float64(longitude_deg))
    lat = np.deg2rad(np.float64(latitude_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = a / np.sqrt(np.float64(1.0) - e2 * sin_lat * sin_lat)
    nph = n + np.float64(height)
    x = nph * cos_lat * np.cos(lon)
    y = nph * cos_lat * np.sin(lon)
    z = (n * (np.float64(1.0) - e2) + np.float64(height)) * sin_lat
    return np.float64(x), np.float64(y), np.float64(z)


def _ecef_to_geodetic(
        x: float,
        y: float,
        z: float,
        a: float,
        e2: float,
) -> tuple[float, float, float]:
    x = np.float64(x)
    y = np.float64(y)
    z = np.float64(z)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    f = np.float64(1.0) - np.sqrt(np.float64(1.0) - e2)
    b = a * (np.float64(1.0) - f)
    if p < 1e-12:
        lat = np.copysign(np.pi / 2.0, z)
        sin_lat = np.sin(lat)
        n = a / np.sqrt(np.float64(1.0) - e2 * sin_lat * sin_lat)
        height = np.abs(z) - n * (np.float64(1.0) - e2)
        return _wrap_longitude(float(np.rad2deg(lon))), float(np.rad2deg(lat)), float(height)

    ep2 = (a * a - b * b) / (b * b)
    theta = np.arctan2(z * a, p * b)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    lat = np.arctan2(z + ep2 * b * sin_th ** 3, p - e2 * a * cos_th ** 3)
    height = np.float64(0.0)
    for _ in range(_MAX_ECEF_ITERS):
        sin_lat = np.sin(lat)
        n = a / np.sqrt(np.float64(1.0) - e2 * sin_lat * sin_lat)
        height = p / np.cos(lat) - n
        lat_new = np.arctan2(z, p * (np.float64(1.0) - e2 * n / (n + height)))
        if abs(float(lat_new - lat)) < _LAT_CONV_RAD:
            lat = lat_new
            break
        lat = lat_new
    sin_lat = np.sin(lat)
    n = a / np.sqrt(np.float64(1.0) - e2 * sin_lat * sin_lat)
    if abs(float(np.cos(lat))) < 1e-12:
        height = z / np.sin(lat) - n * (np.float64(1.0) - e2)
    else:
        height = p / np.cos(lat) - n
    return _wrap_longitude(float(np.rad2deg(lon))), float(np.rad2deg(lat)), float(height)


def _ecef_to_enu(
        dx: float,
        dy: float,
        dz: float,
        origin_lon_deg: float,
        origin_lat_deg: float,
) -> tuple[np.float64, np.float64, np.float64]:
    lon = np.deg2rad(np.float64(origin_lon_deg))
    lat = np.deg2rad(np.float64(origin_lat_deg))
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return np.float64(east), np.float64(north), np.float64(up)


def _enu_to_ecef(
        east: float,
        north: float,
        up: float,
        origin_lon_deg: float,
        origin_lat_deg: float,
) -> tuple[np.float64, np.float64, np.float64]:
    lon = np.deg2rad(np.float64(origin_lon_deg))
    lat = np.deg2rad(np.float64(origin_lat_deg))
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up
    return np.float64(dx), np.float64(dy), np.float64(dz)


def _enu_to_local(east: float, north: float, up: float, bearing_deg: float) -> tuple[float, float, float]:
    theta = np.deg2rad(np.float64(bearing_deg))
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    x = east * sin_t + north * cos_t
    y = -east * cos_t + north * sin_t
    return float(x), float(y), float(up)


def _local_to_enu(x: float, y: float, z: float, bearing_deg: float) -> tuple[float, float, float]:
    theta = np.deg2rad(np.float64(bearing_deg))
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    east = x * sin_t - y * cos_t
    north = x * cos_t + y * sin_t
    return float(east), float(north), float(z)


def _normalize_earth_mapping(kvs: dict[str, Any]) -> dict[str, float]:
    longitude = kvs.get("longitude", kvs.get("lon"))
    latitude = kvs.get("latitude", kvs.get("lat"))
    height = kvs.get("ellipsoidal_height", kvs.get("alt", 0.0))
    if longitude is None or latitude is None:
        raise ValueError("earth frame requires longitude/latitude or lon/lat")
    return {
        "longitude": _require_finite("longitude", longitude),
        "latitude": _require_finite("latitude", latitude),
        "ellipsoidal_height": _require_finite("ellipsoidal_height", height),
    }


def _require_origin(origin: Any, expected: type, method: str) -> None:
    if type(origin) is not expected:
        raise CoordinateFrameException(
            detail=(
                f"{method} origin must be {expected.__name__}, "
                f"got {type(origin).__name__}"
            ),
        )


def _install_earth_aliases(cls):
    original_init = cls.__init__

    def __init__(
            self,
            longitude: float | None = None,
            latitude: float | None = None,
            ellipsoidal_height: float | None = None,
            *,
            lon: float | None = None,
            lat: float | None = None,
            alt: float | None = None,
    ):
        if longitude is None:
            longitude = lon
        if latitude is None:
            latitude = lat
        if ellipsoidal_height is None:
            ellipsoidal_height = 0.0 if alt is None else alt
        if longitude is None or latitude is None:
            raise TypeError(f"{cls.__name__} requires longitude/latitude or lon/lat")
        original_init(self, float(longitude), float(latitude), float(ellipsoidal_height))

    cls.__init__ = __init__
    return cls


def _earth_to_local(frame: Any, origin: Any, x_axis_bearing_deg: float) -> LocalFrame:
    bearing = _require_finite("x_axis_bearing_deg", x_axis_bearing_deg)
    ox, oy, oz = origin._to_ecef()
    x, y, z = frame._to_ecef()
    east, north, up = _ecef_to_enu(
        float(x - ox),
        float(y - oy),
        float(z - oz),
        origin.longitude,
        origin.latitude,
    )
    lx, ly, lz = _enu_to_local(float(east), float(north), float(up), bearing)
    return LocalFrame(x=lx, y=ly, z=lz)


def _local_to_earth(local: LocalFrame, origin: Any, x_axis_bearing_deg: float, earth_cls: type):
    bearing = _require_finite("x_axis_bearing_deg", x_axis_bearing_deg)
    local._validate()
    east, north, up = _local_to_enu(local.x, local.y, local.z, bearing)
    ox, oy, oz = origin._to_ecef()
    dx, dy, dz = _enu_to_ecef(east, north, up, origin.longitude, origin.latitude)
    lon, lat, height = _ecef_to_geodetic(
        float(ox + dx),
        float(oy + dy),
        float(oz + dz),
        origin._a,
        origin._e2,
    )
    return earth_cls(longitude=lon, latitude=lat, ellipsoidal_height=height)


class _EarthMixin(DataClassJsonMixin):
    longitude: float
    latitude: float
    ellipsoidal_height: float
    _a: ClassVar[float]
    _e2: ClassVar[float]

    def _validate(self) -> None:
        self.longitude = _require_finite("longitude", self.longitude)
        self.latitude = _require_finite("latitude", self.latitude)
        self.ellipsoidal_height = _require_finite(
            "ellipsoidal_height",
            self.ellipsoidal_height,
        )
        _validate_lon_lat(self.longitude, self.latitude)

    def to_dict(self, encode_json: bool = False) -> dict[str, float]:
        del encode_json
        return {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "ellipsoidal_height": self.ellipsoidal_height,
        }

    def _to_ecef(self) -> tuple[np.float64, np.float64, np.float64]:
        return _geodetic_to_ecef(
            self.longitude,
            self.latitude,
            self.ellipsoidal_height,
            self._a,
            self._e2,
        )

    @classmethod
    def from_dict(cls, kvs: dict, *, infer_missing: bool = False):
        del infer_missing
        return cls(**_normalize_earth_mapping(kvs))


@_install_earth_aliases
@dataclass
class WGS84Frame(_EarthMixin):
    longitude: float
    latitude: float
    ellipsoidal_height: float = 0.0
    _a: ClassVar[float] = WGS84_A
    _e2: ClassVar[float] = WGS84_E2

    def __post_init__(self) -> None:
        self._validate()

    @property
    def lon(self) -> float:
        return self.longitude

    @property
    def lat(self) -> float:
        return self.latitude

    @property
    def alt(self) -> float:
        return self.ellipsoidal_height

    def __repr__(self) -> str:
        return (
            f"lon: {self.longitude:.4f}, lat: {self.latitude:.4f}, "
            f"alt: {self.ellipsoidal_height:.4f}"
        )

    def to_cgcs2000(self) -> CGCS2000Frame:
        x, y, z = self._to_ecef()
        lon, lat, height = _ecef_to_geodetic(x, y, z, CGCS2000_A, CGCS2000_E2)
        return CGCS2000Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)

    def to_local(self, *, origin: WGS84Frame, x_axis_bearing_deg: float = 90.0) -> LocalFrame:
        _require_origin(origin, WGS84Frame, "WGS84Frame.to_local")
        return _earth_to_local(self, origin, x_axis_bearing_deg)

    def to_xyz(self) -> LocalFrame:
        """存量 ECEF 换算，保持历史公式与常量，供 GeoPoint 兼容调用。"""
        earthe = 0.0818191908426
        radius_earth = 6378140.0
        omge = 0.99330562000987
        pi = 3.14159265358979
        deg2rad = pi / 180.0
        lat = self.latitude * deg2rad
        lon = self.longitude * deg2rad
        d = earthe * math.sin(lat)
        n = radius_earth / math.sqrt(1 - math.pow(d, 2))
        nph = n + self.ellipsoidal_height
        x = nph * math.cos(lat) * math.cos(lon)
        y = nph * math.cos(lat) * math.sin(lon)
        z = (omge * n + self.ellipsoidal_height) * math.sin(lat)
        return LocalFrame(x=x, y=y, z=z)

    @staticmethod
    def distance(p1: WGS84Frame, p2: WGS84Frame) -> float:
        return LocalFrame.distance(p1.to_xyz(), p2.to_xyz())

    @staticmethod
    def central_angle(p1: WGS84Frame, p2: WGS84Frame) -> float:
        lon1_rad = math.radians(p1.longitude)
        lat1_rad = math.radians(p1.latitude)
        lon2_rad = math.radians(p2.longitude)
        lat2_rad = math.radians(p2.latitude)
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        return 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def angle(p1: WGS84Frame, p2: WGS84Frame, p3: WGS84Frame) -> float:
        ab = WGS84Frame.central_angle(p1, p2)
        ac = WGS84Frame.central_angle(p2, p3)
        bc = WGS84Frame.central_angle(p1, p3)
        numerator = math.cos(ac) - math.cos(ab) * math.cos(bc)
        denominator = math.sin(ab) * math.sin(bc)
        if denominator == 0:
            return 0
        cos_angle = numerator / denominator
        cos_angle = np.clip(cos_angle, -1, 1)
        return math.degrees(math.acos(cos_angle))


@_install_earth_aliases
@dataclass
class CGCS2000Frame(_EarthMixin):
    longitude: float
    latitude: float
    ellipsoidal_height: float = 0.0
    _a: ClassVar[float] = CGCS2000_A
    _e2: ClassVar[float] = CGCS2000_E2

    def __post_init__(self) -> None:
        self._validate()

    def to_wgs84(self) -> WGS84Frame:
        x, y, z = self._to_ecef()
        lon, lat, height = _ecef_to_geodetic(x, y, z, WGS84_A, WGS84_E2)
        return WGS84Frame(longitude=lon, latitude=lat, ellipsoidal_height=height)

    def to_local(
            self,
            *,
            origin: CGCS2000Frame,
            x_axis_bearing_deg: float = 90.0,
    ) -> LocalFrame:
        _require_origin(origin, CGCS2000Frame, "CGCS2000Frame.to_local")
        return _earth_to_local(self, origin, x_axis_bearing_deg)


@dataclass
class LocalFrame(DataClassJsonMixin):
    x: float
    y: float
    z: float = 0.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.x = _require_finite("x", self.x)
        self.y = _require_finite("y", self.y)
        self.z = _require_finite("z", self.z)

    @classmethod
    def from_dict(cls, kvs: dict, *, infer_missing: bool = False) -> LocalFrame:
        del infer_missing
        if "x" not in kvs or "y" not in kvs:
            raise ValueError("LocalFrame requires x and y")
        return cls(
            x=_require_finite("x", kvs["x"]),
            y=_require_finite("y", kvs["y"]),
            z=_require_finite("z", kvs.get("z", 0.0)),
        )

    def to_dict(self, encode_json: bool = False) -> dict[str, float]:
        del encode_json
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_wgs84(self, *, origin: WGS84Frame, x_axis_bearing_deg: float = 90.0) -> WGS84Frame:
        _require_origin(origin, WGS84Frame, "LocalFrame.to_wgs84")
        return _local_to_earth(self, origin, x_axis_bearing_deg, WGS84Frame)

    def to_cgcs2000(
            self,
            *,
            origin: CGCS2000Frame,
            x_axis_bearing_deg: float = 90.0,
    ) -> CGCS2000Frame:
        _require_origin(origin, CGCS2000Frame, "LocalFrame.to_cgcs2000")
        return _local_to_earth(self, origin, x_axis_bearing_deg, CGCS2000Frame)

    @staticmethod
    def distance(p1: LocalFrame, p2: LocalFrame) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


GeoPoint = WGS84Frame
XYZPosition = LocalFrame


@dataclass
class JamDirection(DataClassJsonMixin):
    azimuth: float
    elevation: float


@dataclass
class Speed(DataClassJsonMixin):
    vx: float
    vy: float
    vz: float


@dataclass
class AzimuthFov(DataClassJsonMixin):
    pass
