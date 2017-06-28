@rem copy models from linux servers to local
@rem usage like: cp_and_test_models.bat 10.150.144.94 e2d2_reslayer_wise_att0_bpe_tc_ft640k_lr0.9 810000 840000 en-fr_bpe_tc

set ip=%1
set modelname=%2
set start=%3
set end=%4
set task=%5

for /l %%x in (%start%, 10000, %end%) do (
pscp -pw fetia fetia@%ip%:/data/users/fetia/DL4NMT_Theano/model/complete/%modelname%.iter%%x.* model\complete
)

set /a startdivide=%start% / 10000
set /a enddivide=%end% / 10000

python seq_translate.py model\complete\%modelname%.npz --start %startdivide% --end %enddivide% --dataset %task% --beam 5
