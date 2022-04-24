echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 0 origin mask

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 0 origin mask

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 0 origin vtype

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 0 origin vtype

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 1 origin mask

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 1 origin mask

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 1 origin vtype

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 1 origin vtype

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 2 origin mask

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 2 origin mask

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_ablation.py train_ $1 2 origin vtype

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_ablation.py test_ $1 2 origin vtype


