ip=$1
modelname=$2
start=$3
end=$4
task=$5
for fid in $(seq ${start} 10000 ${end})
do
    sshpass -p 'fetia' scp fetia@${ip}:~/fetia/DL4NMT_Theano/model/complete/${modelname}.iter${fid}.* model/complete
done
startdivide=$(expr ${start} / 10000)
enddivide=$(expr ${end} / 10000)
python seq_translate.py model/complete/${modelname}.npz --start ${startdivide} --end ${enddivide} --dataset ${task} --beam 6
