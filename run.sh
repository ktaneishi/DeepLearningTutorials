#!/bin/sh

DATA_SET=cpi.npz

if [ ! -f $DATA_SET ]; then
    echo "Can't find dataset for DBN-Kyoto (cpi.npz) in current directory"
    echo "Please get the it from syncplicity via below link"
    echo "\thttps://my.syncplicity.com/share/vvks9oqxas1xneg/cpi"
    exit
fi

[ ! -d result ] && mkdir result
python benchmark.py $DATA_SET 2000 3
