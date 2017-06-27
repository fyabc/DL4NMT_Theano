@rem copy models from linux servers to local

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
