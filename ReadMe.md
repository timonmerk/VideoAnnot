## Automatic video annotation Mediapipe

Based on this tutorial: https://google.github.io/mediapipe/solutions/pose.html

Writes out .csv with annotated keypoints for each frame.

Install new conda environment via:
```
conda env create --file=env.yml --name mediapipe_annot --user
```

Run with activated conda environment via:
```
python main.py MOVE_NAME True
```

The second boolean argument specifies if the annotated keypoints are plotted during the keypoint extraction.