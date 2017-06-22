@rem Make truecase dataset.
@rem Usage: make_truecase.bat train1 train2 valid1 valid2 small1 small2 [single_dict]

set train1=%1
set train2=%2
set valid1=%3
set valid2=%4
set test1=%5
set test2=%6
set small1=%7
set small2=%8
set single_dict=%9

set model1=data/train/%1.model
set model2=data/train/%2.model

@rem Build truecase model.
perl train-truecaser.perl --model %model1% --corpus data/train/%train1%
perl train-truecaser.perl --model %model2% --corpus data/train/%train2%

@rem Truecase training data.
perl truecase.perl --model %model1% < data/train/%train1% > data/train/tc_%train1%
perl truecase.perl --model %model2% < data/train/%train2% > data/train/tc_%train2%

@rem Truecase small training data.
perl truecase.perl --model %model1% < data/train/%small1% > data/train/tc_%small1%
perl truecase.perl --model %model2% < data/train/%small2% > data/train/tc_%small2%

@rem Truecase validation data.
perl truecase.perl --model %model1% < data/dev/%valid1% > data/dev/tc_%valid1%
perl truecase.perl --model %model2% < data/dev/%valid2% > data/dev/tc_%valid2%

@rem Truecase test data.
perl truecase.perl --model %model1% < data/test/%test1% > data/test/tc_%test1%
perl truecase.perl --model %model2% < data/test/%test2% > data/test/tc_%test2%

@rem Extract new dictionary.
if "%single_dict%" == "" (
    python scripts/build_dictionary.py tc_%train1%
    python scripts/build_dictionary.py tc_%train2%
) else (
    python scripts/build_dictionary.py tc_%train1% tc_%train2% -o %single_dict%
)
