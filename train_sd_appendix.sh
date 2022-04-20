echo "====== train ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 1 origin MIDA

echo "====== test ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 1 origin MIDA

echo "====== train ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 2 origin GAIN

echo "====== test ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 2 origin GAIN

echo "====== train ip ? ? MICE ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 2 origin MICE

echo "====== test ip ? ? MICE ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 2 origin MICE

echo "====== train ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 2 origin MIDA

echo "====== test ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 2 origin MIDA
