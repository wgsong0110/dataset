import numpy as np
import torch
import json
import struct
import math
from pathlib import Path
from typing import Union

from .camera import Camera


class CameraPool:
    def __init__(self, rst: str, dataset: str, scene: str, split: str = "train",
                 coordinate_system: str = "colmap", width=None, height=None):
        self.rst = rst
        self.dataset = dataset
        self.scene = scene
        self.split = split
        self.coordinate_system = coordinate_system.lower()
        self.width = width
        self.height = height

        if self.coordinate_system not in ["colmap", "blender", "opengl"]:
            raise ValueError(
                f"Invalid coordinate_system '{coordinate_system}'. "
                "Must be 'colmap', 'blender', or 'opengl'"
            )

        # Construct base path
        self.base_path = Path(rst) / dataset / scene

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

        self.cameras_data = {}  # For COLMAP: camera_id -> camera params
        self.images_data = {}   # For COLMAP: image_name -> camera data
        self.image_names = []   # Ordered list of image names

        # Determine data source and load
        self.data_source = self._determine_data_source()
        self._load_data()

    def _determine_data_source(self) -> str:
        # Check for JSON files first (NeRF synthetic 포맷 우선)
        json_candidates = [
            self.base_path / "camera" / f"transforms_{self.split}.json",
            self.base_path / "camera" / "transforms.json",
        ]

        for json_path in json_candidates:
            if json_path.exists():
                self.json_path = json_path
                return "json"

        # Check for COLMAP data (새로운 구조만)
        colmap_path = self.base_path / "camera" / "0"
        if (colmap_path / "cameras.bin").exists() or (colmap_path / "cameras.txt").exists():
            self.colmap_path = colmap_path
            return "colmap"

        raise FileNotFoundError(
            f"Could not find COLMAP or JSON data in {self.base_path}/camera/"
        )

    def _load_data(self):
        if self.data_source == "colmap":
            self._load_colmap_data()
        else:
            self._load_json_data()

    def _load_colmap_data(self):
        if (self.colmap_path / "cameras.bin").exists():
            self._read_cameras_binary()
            self._read_images_binary()
        elif (self.colmap_path / "cameras.txt").exists():
            self._read_cameras_text()
            self._read_images_text()
        else:
            raise FileNotFoundError(
                f"Could not find cameras file in {self.colmap_path}"
            )

        # Create ordered list of image names and apply LLFF hold for train/test split
        all_image_names = sorted(self.images_data.keys())

        # LLFF hold: 8번째마다 test
        llffhold = 8
        test_indices = set(idx for idx in range(len(all_image_names)) if idx % llffhold == 0)

        if self.split == "train":
            self.image_names = [name for idx, name in enumerate(all_image_names) if idx not in test_indices]
        elif self.split == "test":
            self.image_names = [name for idx, name in enumerate(all_image_names) if idx in test_indices]
        else:
            # val or other splits: use all images
            self.image_names = all_image_names

    def _read_cameras_text(self):
        cameras_file = self.colmap_path / "cameras.txt"
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = np.array([float(x) for x in parts[4:]])

                self.cameras_data[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }

    def _read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def _get_camera_params_count(self, model_id):
        camera_model_num_params = {
            0: 3,   # SIMPLE_PINHOLE
            1: 4,   # PINHOLE
            2: 4,   # SIMPLE_RADIAL
            3: 5,   # RADIAL
            4: 8,   # OPENCV
            5: 8,   # OPENCV_FISHEYE
            6: 12,  # FULL_OPENCV
            7: 5,   # FOV
            8: 4,   # SIMPLE_RADIAL_FISHEYE
            9: 5,   # RADIAL_FISHEYE
            10: 12, # THIN_PRISM_FISHEYE
        }
        return camera_model_num_params.get(model_id, 0)

    def _read_cameras_binary(self):
        cameras_file = self.colmap_path / "cameras.bin"

        self.cameras_data = {}
        with open(cameras_file, 'rb') as f:
            num_cameras = self._read_next_bytes(f, 8, "Q")[0]

            for _ in range(num_cameras):
                # 24 bytes를 한 번에 읽음: camera_id(4) + model_id(4) + width(8) + height(8)
                camera_properties = self._read_next_bytes(f, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]

                # Get number of parameters based on model_id
                num_params = self._get_camera_params_count(model_id)

                # Read camera parameters
                params = self._read_next_bytes(f, num_bytes=8 * num_params, format_char_sequence="d" * num_params)

                model_names = {
                    0: 'SIMPLE_PINHOLE', 1: 'PINHOLE', 2: 'SIMPLE_RADIAL',
                    3: 'RADIAL', 4: 'OPENCV', 5: 'OPENCV_FISHEYE',
                    6: 'FULL_OPENCV', 7: 'FOV', 8: 'SIMPLE_RADIAL_FISHEYE',
                    9: 'RADIAL_FISHEYE', 10: 'THIN_PRISM_FISHEYE'
                }
                model = model_names.get(model_id, f'UNKNOWN_{model_id}')

                self.cameras_data[camera_id] = {
                    'model': model,
                    'width': int(width),
                    'height': int(height),
                    'params': np.array(params)
                }

    def _read_images_text(self):
        images_file = self.colmap_path / "images.txt"
        with open(images_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue

            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]

            # Filter by split and process image name
            processed_name = self._process_image_name(image_name)
            if processed_name is None:
                i += 2
                continue

            # Convert quaternion to rotation matrix
            R = self._qvec2rotmat([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])

            self.images_data[processed_name] = {
                'camera_id': camera_id,
                'R': R,
                't': t,
                'image_id': image_id
            }

            i += 2  # Skip the points2D line

    def _read_images_binary(self):
        images_file = self.colmap_path / "images.bin"
        with open(images_file, 'rb') as f:
            num_images = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack('I', f.read(4))[0]
                qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
                tx, ty, tz = struct.unpack('ddd', f.read(24))
                camera_id = struct.unpack('I', f.read(4))[0]

                # Read image name
                image_name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    image_name += char
                image_name = image_name.decode('utf-8')

                # Skip points2D
                num_points2D = struct.unpack('Q', f.read(8))[0]
                f.read(24 * num_points2D)  # Each point2D is 24 bytes

                # Filter by split and process image name
                processed_name = self._process_image_name(image_name)
                if processed_name is None:
                    continue

                # Convert quaternion to rotation matrix
                # R = self._qvec2rotmat([qw, qx, qy, qz])
                R = np.transpose(qvec2rotmat([qw, qx, qy, qz]))
                t = np.array([tx, ty, tz])
                self.images_data[processed_name] = {
                    'camera_id': camera_id,
                    'R': R,
                    't': t,
                    'image_id': image_id
                }

    def _process_image_name(self, image_name: str):
        # LLFF hold 방식을 사용하므로 이미지 이름은 그대로 반환
        return image_name

    @staticmethod
    def _qvec2rotmat(qvec):
        qvec = np.array(qvec)
        w, x, y, z = qvec
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R

    def _build_intrinsic_matrix(self, camera_params: dict) -> np.ndarray:
        params = camera_params['params']
        model = camera_params['model']

        if model == 'SIMPLE_PINHOLE':
            # params: f, cx, cy
            f, cx, cy = params[0], params[1], params[2]
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ])
        elif model == 'PINHOLE':
            # params: fx, fy, cx, cy
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        elif model in ['SIMPLE_RADIAL', 'RADIAL', 'OPENCV']:
            # params: fx, (fy,) cx, cy, (distortion params...)
            # For now, just use the focal length and principal point
            if model == 'SIMPLE_RADIAL':
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Unsupported camera model: {model}")

        return K

    def _build_extrinsic_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t
        return extrinsics

    def _get_camera_from_colmap(self, image_name: str) -> Camera:
        if image_name not in self.images_data:
            raise KeyError(f"Image '{image_name}' not found in COLMAP data")

        image_data = self.images_data[image_name]
        camera_id = image_data['camera_id']

        if camera_id not in self.cameras_data:
            raise KeyError(f"Camera ID {camera_id} not found in COLMAP data")

        camera_params = self.cameras_data[camera_id]

        # Build intrinsic matrix
        intrinsics = self._build_intrinsic_matrix(camera_params)

        # Extract focal lengths from intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        # Use constructor-provided dimensions if available, otherwise use COLMAP dimensions
        orig_width = camera_params['width']
        orig_height = camera_params['height']

        if self.width is not None and self.height is not None:
            width = self.width
            height = self.height
            # Adjust focal lengths for resolution scaling
            scale_x = width / orig_width
            scale_y = height / orig_height
            fx = fx * scale_x
            fy = fy * scale_y
        else:
            width = orig_width
            height = orig_height

        # Calculate field of view
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))

        return Camera(
            R=image_data['R'], T=image_data['t'],
            FoVy=FoVy,
            height=height,
            width=width,
            FoVx=FoVx)

    def _load_json_data(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, 'r') as f:
            self.json_data = json.load(f)

        # Parse camera intrinsics
        self._parse_intrinsics()

        # Parse frames
        self._parse_frames()

    def _parse_intrinsics(self):
        # Use constructor-provided dimensions first, then JSON, then default
        if self.width is None:
            self.width = self.json_data.get('w', 800)
        if self.height is None:
            self.height = self.json_data.get('h', 800)

        # Get focal lengths
        if 'fl_x' in self.json_data and 'fl_y' in self.json_data:
            self.fl_x = self.json_data['fl_x']
            self.fl_y = self.json_data['fl_y']
        elif 'camera_angle_x' in self.json_data:
            # Calculate focal length from field of view
            camera_angle_x = self.json_data['camera_angle_x']
            self.fl_x = 0.5 * self.width / np.tan(0.5 * camera_angle_x)
            self.fl_y = self.fl_x

            # If camera_angle_y is provided, use it
            if 'camera_angle_y' in self.json_data:
                camera_angle_y = self.json_data['camera_angle_y']
                self.fl_y = 0.5 * self.height / np.tan(0.5 * camera_angle_y)
        else:
            raise ValueError(
                "JSON must contain either 'fl_x'/'fl_y' or 'camera_angle_x'"
            )

        # Get principal point (default to image center)
        self.cx = self.json_data.get('cx', self.width / 2.0)
        self.cy = self.json_data.get('cy', self.height / 2.0)

        # Build intrinsic matrix
        self.json_intrinsics = np.array([
            [self.fl_x, 0, self.cx],
            [0, self.fl_y, self.cy],
            [0, 0, 1]
        ])

    def _parse_frames(self):
        if 'frames' not in self.json_data:
            raise ValueError("JSON must contain 'frames' field")

        self.json_cameras = {}  # image_name -> Camera

        # Calculate field of view from focal lengths
        FoVx = 2 * math.atan(self.width / (2 * self.fl_x))
        FoVy = 2 * math.atan(self.height / (2 * self.fl_y))

        for frame in self.json_data['frames']:
            if 'file_path' not in frame:
                raise ValueError("Each frame must contain 'file_path'")

            if 'transform_matrix' not in frame:
                raise ValueError("Each frame must contain 'transform_matrix'")

            file_path = frame['file_path']
            image_name = Path(file_path).name
            camera_angle_y = self.json_data.get('camera_angle_y', None)
            camera = parse_camera_from_transform(frame['transform_matrix'], self.width, self.height, self.json_data['camera_angle_x'], camera_angle_y=camera_angle_y)

            if image_name not in self.json_cameras:
                self.image_names.append(image_name)
            self.json_cameras[image_name] = camera
            self.json_cameras[file_path] = camera

    def _get_camera_from_json(self, image_name: str) -> Camera:
        if image_name not in self.json_cameras:
            raise KeyError(
                f"Image '{image_name}' not found in JSON data. "
                f"Available images: {self.image_names}"
            )

        return self.json_cameras[image_name]

    def __getitem__(self, key: Union[int, str]) -> Camera:
        if isinstance(key, int):
            # Access by index
            if key < 0 or key >= len(self.image_names):
                raise IndexError(f"Index {key} out of range [0, {len(self.image_names)})")
            image_name = self.image_names[key]
        elif isinstance(key, str):
            # Access by image name
            image_name = key
        else:
            raise TypeError(f"Key must be int or str, not {type(key)}")

        return self.get_camera(image_name)

    def get_camera(self, image_name: str) -> Camera:
        if self.data_source == "colmap":
            return self._get_camera_from_colmap(image_name)
        else:
            return self._get_camera_from_json(image_name)

    def get_all_image_names(self):
        return self.image_names.copy()

    def __len__(self):
        return len(self.image_names)

    def __contains__(self, image_name: str):
        if self.data_source == "colmap":
            return image_name in self.images_data
        else:
            return image_name in self.json_cameras

    def __repr__(self):
        repr_str = (f"CameraPool(rst='{self.rst}', dataset='{self.dataset}', "
                    f"scene='{self.scene}', split='{self.split}', "
                    f"source='{self.data_source}'")

        # Show coordinate system for JSON sources
        if self.data_source == "json":
            repr_str += f", coord_sys='{self.coordinate_system}'"

        repr_str += f", num_cameras={len(self)})"
        return repr_str


# NeRF transform matrix로부터 Camera 객체 생성
def parse_camera_from_transform(transform_matrix, width: int, height: int,
                                camera_angle_x: float, device: str = "cuda", camera_angle_y: float = None,
                                apply_coord_transform: bool = True):
    import torch, math
    from .camera import Camera

    # Camera-to-world matrix
    c2w = torch.tensor(transform_matrix, dtype=torch.float32, device=device)

    # OpenGL/Blender (Y up, Z back) → COLMAP (Y down, Z forward) 변환
    # Blender synthetic 데이터만 변환 필요, COLMAP 기반 데이터는 변환 불필요
    if apply_coord_transform:
        c2w[:3, 1:3] *= -1

    w2c = torch.inverse(c2w)

    # Rotation 및 translation 추출
    R = w2c[:3, :3].transpose(0, 1).cpu().numpy()
    T = w2c[:3, 3].cpu().numpy()

    # FoV 계산
    FoVx = camera_angle_x
    if camera_angle_y is not None:
        FoVy = camera_angle_y
    else:
        FoVy = 2 * math.atan(math.tan(FoVx / 2) * height / width)

    # Camera 객체 생성
    camera = Camera(R=R, T=T, FoVy=FoVy, height=height, width=width, data_device=device, FoVx=FoVx)
    return camera

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])