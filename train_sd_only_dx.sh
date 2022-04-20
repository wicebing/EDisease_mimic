echo "====== train ip ? ? mean ======" $1 
python EDisease_train_mimic.py train_mlp $1 0 origin only_dx

echo "====== test ip ? ? mean ======" $1 
python EDisease_train_mimic.py test_mlp $1 0 origin only_dx

echo "====== train ip ? ? mean ======" $1 
python EDisease_train_mimic.py train_mlp $1 1 origin only_dx

echo "====== test ip ? ? mean ======" $1 
python EDisease_train_mimic.py test_mlp $1 1 origin only_dx

echo "====== train ip ? ? mean ======" $1 
python EDisease_train_mimic.py train_mlp $1 2 origin only_dx

echo "====== test ip ? ? mean ======" $1 
python EDisease_train_mimic.py test_mlp $1 2 origin only_dx
