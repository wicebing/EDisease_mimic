echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_timesequence_ablation.py train_ $1 $2 origin mask

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_timesequence_ablation.py test_ $1 $2 origin vtype





