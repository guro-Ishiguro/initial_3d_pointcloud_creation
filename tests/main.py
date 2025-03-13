from integrate_3d_pointcloud import PointCloudIntegrator
import argparse
import logging


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Show viewer during point cloud generation.",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record viewer output to video."
    )
    args = parser.parse_args()

    integrator = PointCloudIntegrator(args)
    integrator.run()


if __name__ == "__main__":
    main()
