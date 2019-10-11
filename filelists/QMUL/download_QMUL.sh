#!/usr/bin/env bash
wget http://www.eecs.qmul.ac.uk/~sgg/QMUL_FaceDataset/QMULFaceDataset.zip
unzip QMULFaceDataset.zip
python write_QMUL_filelist.py
