#cd preprocess;
#python3 precompute_PanjivaRecordID_hdf_v1.py;
#cd ..;
python3 processor_1.py --dir china_import;
python3 processor_1.py --dir china_export;
python3 processor_1.py --dir us_import;
python3 processor_1.py --dir peru_export;

python3 addTextFlags_toResults.py --dir china_import;
python3 addTextFlags_toResults.py --dir china_export;
python3 addTextFlags_toResults.py --dir us_import;
python3 addTextFlags_toResults.py --dir peru_export;