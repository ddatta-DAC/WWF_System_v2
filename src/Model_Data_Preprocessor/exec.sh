
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

