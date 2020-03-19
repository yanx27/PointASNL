cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../

cd tf_ops/
sh compile_ops.sh
