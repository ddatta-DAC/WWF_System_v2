cd DataPreprocess;
python3 define_segments.py;
python3 clean_data_v0.py;
python3 model_data_clean_v0.py;
cd ..;

cd Model_Data_Preprocessor;

for i in $(seq 1 10)
do
python3 data_generator.py --dir us_import --case ${i}
done

for i in $(seq 1 10)
do
python3 data_generator.py --dir china_export --case ${i}
done

for i in $(seq 1 9)
do
python3 data_generator.py --dir china_import --case ${i}
done

for i in $(seq 1 2)
do
python3 data_generator.py --dir peru_export --case ${i}
done

cd ..;
