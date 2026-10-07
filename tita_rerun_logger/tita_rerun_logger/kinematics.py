"""Minimal URDF model: masses, meshes, forward kinematics and CoM (numpy only, no ROS)."""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_xyzw_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_xyzw_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def yaw_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axis_angle(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def cross(a, b):
    """3-vector cross product (np.cross is ~10x slower on single vectors)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def origin_to_T(elem):
    """URDF <origin xyz rpy> element (or None) -> 4x4 transform."""
    if elem is None:
        return np.eye(4)
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    return make_T(rpy_to_matrix(*rpy), xyz)


class UrdfModel:
    """Links (mass, CoM, meshes) and joints of a URDF, with FK from a floating base.

    mesh_resolver(filename) -> absolute path or None; used only for the 3D view.
    """

    def __init__(self, xml_string, mesh_resolver=None):
        self.links = {}            # name -> dict(mass, com (3,), visuals [(path, T, scale)])
        self.joints_by_child = {}  # child -> dict(name, parent, type, T, axis)
        self.children = {}         # parent -> [child links]
        self.effort_limits = {}    # joint name -> URDF effort limit
        self.total_mass = 0.0
        self._resolve = mesh_resolver or (lambda f: f if os.path.isabs(f) else None)
        self._parse(xml_string)
        roots = [n for n in self.links if n not in self.joints_by_child]
        self.root = roots[0] if roots else None

    def _parse(self, xml_string):
        root = ET.fromstring(xml_string)
        for link in root.findall("link"):
            name = link.get("name")
            entry = {"mass": 0.0, "com": np.zeros(3), "visuals": []}
            inertial = link.find("inertial")
            if inertial is not None:
                mass_el = inertial.find("mass")
                if mass_el is not None:
                    entry["mass"] = float(mass_el.get("value", "0"))
                entry["com"] = origin_to_T(inertial.find("origin"))[:3, 3]
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    continue
                path = self._resolve(mesh.get("filename", ""))
                if path is None or not os.path.isfile(path):
                    continue
                scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
                entry["visuals"].append((path, origin_to_T(visual.find("origin")), scale))
            self.links[name] = entry
            self.total_mass += entry["mass"]

        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            parent = joint.find("parent").get("link")
            axis_el = joint.find("axis")
            axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None
                                                else "1 0 0").split()])
            n = np.linalg.norm(axis)
            self.joints_by_child[child] = {
                "name": joint.get("name"),
                "parent": parent,
                "type": joint.get("type"),
                "T": origin_to_T(joint.find("origin")),
                "axis": axis / n if n > 0 else axis,
            }
            self.children.setdefault(parent, []).append(child)
            limit = joint.find("limit")
            if limit is not None and limit.get("effort") is not None:
                eff = float(limit.get("effort"))
                if eff > 0:
                    self.effort_limits[joint.get("name")] = eff

    # ------------------------------------------------------------ Gazebo
    def anchor_of(self, link, available):
        """Walk up fixed joints until a link in `available` is found.

        Gazebo lumps links connected by fixed joints into their parent, so they
        do not appear in /link_states. Returns (anchor_link, T_anchor_link) or None.
        """
        T = np.eye(4)
        cur = link
        while cur not in available:
            j = self.joints_by_child.get(cur)
            if j is None or j["type"] != "fixed":
                return None
            T = j["T"] @ T
            cur = j["parent"]
        return cur, T

    # ------------------------------------------------------------ FK
    def forward(self, T_base, v_base, w_base, q, qd):
        """Floating-base FK.

        T_base: 4x4 world pose of the root link; v_base, w_base: world linear/angular
        velocity of the root origin; q, qd: dict joint name -> position/velocity
        (missing joints count as 0). Returns (link_T, link_v, link_w): world pose,
        world linear velocity of the link origin and world angular velocity per link.
        """
        link_T = {self.root: T_base}
        link_v = {self.root: np.asarray(v_base, dtype=float)}
        link_w = {self.root: np.asarray(w_base, dtype=float)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            Tp, vp, wp = link_T[parent], link_v[parent], link_w[parent]
            for child in self.children.get(parent, ()):
                j = self.joints_by_child[child]
                Tj = Tp @ j["T"]
                v = vp + cross(wp, Tj[:3, 3] - Tp[:3, 3])
                w = wp
                if j["type"] in ("revolute", "continuous"):
                    a = j["axis"]
                    Tj = Tj @ make_T(axis_angle(a, q.get(j["name"], 0.0)))
                    w = wp + Tj[:3, :3] @ a * qd.get(j["name"], 0.0)
                elif j["type"] == "prismatic":
                    a = j["axis"]
                    Tj = Tj @ make_T(p=a * q.get(j["name"], 0.0))
                    v = v + Tj[:3, :3] @ a * qd.get(j["name"], 0.0)
                link_T[child], link_v[child], link_w[child] = Tj, v, w
                stack.append(child)
        return link_T, link_v, link_w

    def com(self, link_T, link_v=None, link_w=None):
        """Whole-body CoM (and its velocity if link_v/link_w are given) from link frames."""
        acc, acc_v, mass = np.zeros(3), np.zeros(3), 0.0
        for link, entry in self.links.items():
            m = entry["mass"]
            if m <= 0 or link not in link_T:
                continue
            T = link_T[link]
            r = T[:3, :3] @ entry["com"]
            acc += m * (T[:3, 3] + r)
            if link_v is not None:
                acc_v += m * (link_v[link] + cross(link_w[link], r))
            mass += m
        if mass <= 0:
            return None, None
        return acc / mass, (acc_v / mass if link_v is not None else None)
