from robosim.scene import motion_bench_scene
from robosim.learning.probmap import continuous_occupancy_map
import plotly.graph_objects as go
import tqdm
import torch

qualitative = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _plot_traces_ee_and_arm_qs(robot_vis, ee_traj, qs_traj, max_timestep: int = 80):
    if len(ee_traj.shape) == 2:
        # is not batched
        ee_traj = ee_traj.reshape(-1, *ee_traj.shape)
        qs_traj = qs_traj.reshape(-1, *qs_traj.shape)

    data = []
    for batch_idx, color in zip(range(ee_traj.shape[0]), qualitative):
        _ee_traj = ee_traj[batch_idx, ...]
        _qs_traj = qs_traj[batch_idx, ...]
        if _ee_traj.shape[0] > max_timestep:
            _ee_traj = _ee_traj[:: int(_ee_traj.shape[0] // max_timestep), ...]
            _qs_traj = _qs_traj[:: int(_qs_traj.shape[0] // max_timestep), ...]

        data.append(
            go.Scatter3d(
                x=_ee_traj[:, 0],
                y=_ee_traj[:, 1],
                z=_ee_traj[:, 2],
                mode="lines",
                name=f"original lasa {batch_idx}",
                line_width=10,
                line_color=color,
            )
        )
        data.extend(
            robot_vis.plot_arms(
                _qs_traj.detach(),
                highlight_end_effector=True,
                showlegend=True,
                name=f"arm {batch_idx}",
                color=color,
            )
        )
    return data


def plot_3d_scene_and_arm(robot_vis, occmap, ee_traj, qs_traj):
    fig = go.Figure()
    fig = continuous_occupancy_map.visualise_model_pred(
        occmap, prob_threshold=0.8, marker_showscale=False, marker_colorscale="viridis"
    )
    fig.add_traces(_plot_traces_ee_and_arm_qs(robot_vis, ee_traj, qs_traj))
    fig.show()


def plot_animation_3d_scene_and_arm(
    robot_vis, occmap, original_xs, list_of_qs_timestep, include_every: int = 1
):
    fig = go.Figure(
        data=_plot_traces_ee_and_arm_qs(
            robot_vis, original_xs, torch.Tensor(list_of_qs_timestep[0])
        ),
    )
    # Frames
    frames = []
    for i, fig_datas in enumerate(
        tqdm.tqdm(list_of_qs_timestep, desc="creating animation...")
    ):
        if i % include_every != 0:
            continue
        frames.append(
            go.Frame(
                data=_plot_traces_ee_and_arm_qs(
                    robot_vis, original_xs, torch.Tensor(fig_datas)
                ),
                name=f"step {i}",
            )
        )

    fig.update(frames=frames)

    def frame_args(duration):
        val = {
            "frame": {
                "duration": duration,
                #    "redraw": False,
            },
            "fromcurrent": True,
            "transition": {"duration": 300, "easing": "quadratic-in-out"},
        }
        if duration > 0:
            val["transition"] = {"duration": 0}
        return val

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(10)],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], autorange=False),
            yaxis=dict(range=[-1.5, 1.5], autorange=False),
            zaxis=dict(range=[-1, 2], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        )
    )

    fig.add_traces(
        continuous_occupancy_map.visualise_model_pred(
            occmap,
            prob_threshold=0.8,
            marker_showscale=False,
            marker_colorscale="viridis",
        ).data
    )

    fig.write_html("out.html")
    fig.show()
