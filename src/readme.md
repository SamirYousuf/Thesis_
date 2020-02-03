create a link to local dataset repository:
```
ln -s /usr/local/courses/lt2318/data/guesswhat/ gw
```


how to run several trainings:

```
nohup sh run.sh > log.txt & 
```

how to visualise the accuracy result of a model (`q+c+s+h`):

```
python3 visualise.py q+c+s+h --acc
```
