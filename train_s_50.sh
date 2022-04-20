echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s.py train_ $1 $2 Less50

echo "====== test SPEC ? ? EM ======" $1 $2 
python EDisease_train_mimic_s.py test_ $1 $2 Less50

echo "====== train ip ? ? mean ======" $1 $2
python EDisease_train_mimic_s.py train_mlp $1 $2 Less50

echo "====== test ip ? ? mean ======" $1 $2
python EDisease_train_mimic_s.py test_mlp $1 $2 Less50

echo "====== train ip ? ? EM ======" $1 $2
python EDisease_train_mimic_s.py train_mlp_ip $1 $2 Less50 EM

echo "====== test ip ? ? EM ======" $1 $2
python EDisease_train_mimic_s.py test_mlp_ip $1 $2 Less50 EM

echo "====== train ip ? ? KNN ======" $1 $2
python EDisease_train_mimic_s.py train_mlp_ip $1 $2 Less50 KNN

echo "====== test ip ? ? KNN ======" $1 $2
python EDisease_train_mimic_s.py test_mlp_ip $1 $2 Less50 KNN

echo "====== train ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic_s.py train_mlp_ip $1 $2 Less50 GAIN

echo "====== test ip ? ? GAIN ======" $1 $2
python EDisease_train_mimic_s.py test_mlp_ip $1 $2 Less50 GAIN

echo "====== train ip ? ? MICE ======" $1 $2
python EDisease_train_mimic_s.py train_mlp_ip $1 $2 Less50 MICE

echo "====== test ip ? ? MICE ======" $1 $2
python EDisease_train_mimic_s.py test_mlp_ip $1 $2 Less50 MICE

echo "====== train ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic_s.py train_mlp_ip $1 $2 Less50 MIDA

echo "====== test ip ? ? MIDA ======" $1 $2
python EDisease_train_mimic_s.py test_mlp_ip $1 $2 Less50 MIDA
