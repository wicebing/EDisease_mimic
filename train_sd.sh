echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic.py train_ $1 $2

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic.py test_ $1 $2

echo "====== train ip ? ? mean ======" $1 $2
python EDisease_train_mimic.py train_mlp $1 $2

echo "====== test ip ? ? mean ======" $1 $2
python EDisease_train_mimic.py test_mlp $1 $2

echo "====== train ip ? ? EM ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 $2 EM

echo "====== test ip ? ? EM ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 $2 EM

echo "====== train ip ? ? KNN ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 $2 KNN

echo "====== test ip ? ? KNN ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 $2 KNN

echo "====== train ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 $2 GAIN

echo "====== test ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 $2 GAIN

echo "====== train ip ? ? MICE ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 $2 MICE

echo "====== test ip ? ? MICE ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 $2 MICE

echo "====== train ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py train_mlp_ip $1 $2 MIDA

echo "====== test ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic.py test_mlp_ip $1 $2 MIDA
