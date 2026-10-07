from glob import glob

from setuptools import setup

package_name = "tita_rerun_logger"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jacopo Tedeschi",
    maintainer_email="jacopotedeschi100@gmail.com",
    description="ROS 2 -> Rerun logger for TITA (ground truth, filter, robot, MPX)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rerun_logger = tita_rerun_logger.rerun_logger_node:main",
            "view_logs = tita_rerun_logger.view_logs:main",
        ],
    },
)
