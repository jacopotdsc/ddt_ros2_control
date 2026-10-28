from glob import glob
from setuptools import find_packages, setup

package_name = "fsm_audio"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/fsm_mp3", glob("config/fsm_mp3/*.mp3")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jacopo Tedeschi",
    maintainer_email="jacopotedeschi100@gmail.com",
    description="Announces RL controller FSM transitions using TITA's audio service.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fsm_audio_node = fsm_audio.fsm_audio_node:main",
        ],
    },
)
