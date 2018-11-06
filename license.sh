for f in */*.py
do
cp header.txt tmp.py
cat $f >> tmp.py
cp tmp.py $f
rm tmp.py
done
