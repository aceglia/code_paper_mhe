import numpy as np
from pyomeca import Markers
import csv


class WriteTrc:
    def __init__(self):
        self.output_file_path = None
        self.input_file_path = None
        self.markers = None
        self.marker_names = None
        self.data_rate = None
        self.cam_rate = None
        self.n_frames = None
        self.start_frame = None
        self.units = None
        self.channels = None
        self.time = None

    def _prepare_trc(self):
        headers = [
            ["PathFileType", 4, "(X/Y/Z)", self.output_file_path],
            [
                "DataRate",
                "CameraRate",
                "NumFrames",
                "NumMarkers",
                "Units",
                "OrigDataRate",
                "OrigDataStartFrame",
                "OrigNumFrames",
            ],
            [
                self.data_rate,
                self.cam_rate,
                self.n_frames,
                len(self.marker_names),
                self.units,
                self.data_rate,
                self.start_frame,
                self.n_frames,
            ],
        ]
        markers_row = ["Frame#", "Time"]
        coord_row = ["", ""]
        empty_row = []
        idx = 0
        for i in range(len(self.marker_names) * 3):
            if i % 3 == 0:
                markers_row.append(self.marker_names[idx])
                idx += 1
            else:
                markers_row.append(None)
        headers.append(markers_row)

        for i in range(len(self.marker_names)):
            name_coord = 0
            while name_coord < 3:
                if name_coord == 0:
                    coord_row.append(f"X{i+1}")
                elif name_coord == 1:
                    coord_row.append(f"Y{i+1}")
                elif name_coord == 2:
                    coord_row.append(f"Z{i+1}")
                name_coord += 1

        headers.append(coord_row)
        headers.append(empty_row)
        return headers

    def write_trc(self):
        if self.input_file_path:
            self._read_c3d()
        headers = self._prepare_trc()
        duration = self.n_frames / self.data_rate
        time = np.around(np.linspace(0, duration, self.n_frames), decimals=2) if self.time is None else self.time
        for frame in range(self.markers.shape[2]):
            row = [frame + 1, time[frame]]
            for i in range(self.markers.shape[1]):
                for j in range(3):
                    row.append(self.markers[j, i, frame])
            headers.append(row)
        with open(self.output_file_path, "w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(headers)

    def _read_c3d(self):
        data = Markers.from_c3d(self.input_file_path, usecols=self.channels)
        self.markers = data.values[:3, :, :]
        self.n_frames = len(data.time.values)
        self.marker_names = data.channel.values.tolist()
        self.data_rate = data.attrs["rate"]
        self.units = data.attrs["units"]
        self.start_frame = data.attrs["first_frame"] + 1
        self.cam_rate = self.data_rate if self.cam_rate is None else self.cam_rate
        self.time = data.time.values


class WriteTrcFromC3d(WriteTrc):
    def __init__(
        self,
        output_file_path,
        c3d_file_path,
        data_rate=None,
        cam_rate=None,
        n_frames=None,
        start_frame=1,
        c3d_channels=None,
    ):
        super(WriteTrcFromC3d, self).__init__()
        self.input_file_path = c3d_file_path
        self.output_file_path = output_file_path
        self.data_rate = data_rate
        self.cam_rate = self.data_rate if cam_rate is None else cam_rate
        self.n_frames = n_frames
        self.start_frame = start_frame
        self.channels = c3d_channels

    def write(self):
        self.write_trc()


class WriteTrcFromMarkersData(WriteTrc):
    def __init__(
        self,
        output_file_path,
        markers=None,
        marker_names=None,
        data_rate=None,
        cam_rate=None,
        n_frames=None,
        start_frame=1,
        units="m",
    ):
        super(WriteTrcFromMarkersData, self).__init__()
        self.output_file_path = output_file_path
        self.markers = markers
        self.marker_names = marker_names
        self.data_rate = data_rate
        self.cam_rate = self.data_rate if cam_rate is None else cam_rate
        self.n_frames = n_frames
        self.start_frame = start_frame
        self.units = units
        self.time = None

    def write(self):
        self.write_trc()


if __name__ == "__main__":
    # outfile_path = "data.trc"
    # infile_path = "data.c3d"
    outfile_path = "data/test_09_12_21/Jules/abd.trc"
    infile_path = "data/test_09_12_21/Jules/abd.c3d"
    WriteTrcFromC3d(output_file_path=outfile_path, c3d_file_path=infile_path).write()
