# bee-wings-venation-matching
Project realized by Sabine Hu and Pierre Guillaume.

Match venation position on bee wings

# Files
```bash
.
├── Data
│   ├── ex01.csv
│   ├── ex01.jpg
│   ├── ...
│   └── TRAIN
│       ├── 004 Osmia lignaria f right 4x.csv
│       ├── 004 Osmia lignaria f right 4x.jpg
│       ├── ...
│       └── ...
├── README.md
├── requirements.txt
└── src
    ├── bee-wings-venation-matching.ipynb
    ├── get_intersections.py
    └── Labelization.ipynb
```

* bee-wings-venation-matching.ipynb: notebook containing the detection method
* get_intersections.py: script contaning the detection method
* Labelization.ipynb: notebook used to add intersections on train dataset

# Usage
```shell
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python src/get_intersections.py <path_to_images>
```

# Links
Video: https://youtu.be/JPXGQXHvh84
