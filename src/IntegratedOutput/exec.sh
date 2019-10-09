cd preprocess;
python3 precompute_PanjivaRecordID_hdf_v1.python3;
cd ..;
python3 processor_1.py --dir china_import;
python3 processor_1.py --dir china_export;
python3 processor_1.py --dir us_import;
python3 processor_1.py --dir peru_export;
