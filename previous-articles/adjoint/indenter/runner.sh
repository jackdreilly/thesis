output=$(find /Users/jdr/Documents/github/AdjointPaper/restart/*tex)
 
OIFS="${IFS}"
NIFS=$'\n'
 
IFS="${NIFS}"
 
for LINE in ${output} ; do
    IFS="${OIFS}"
 
    ./latexindent.pl -w ${LINE}
 
    IFS="${NIFS}"
done
IFS="${OIFS}"