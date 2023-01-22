pushd class_notes
rm *.aux *.bbl *.blg *.dvi *.log
for FILE in *.tex; do 
	pdflatex $FILE; 
	bibtex "${FILE%.*}" ;
	pdflatex $FILE; 
	pdflatex $FILE; 
done
rm *.aux *.bbl *.blg *.dvi *.log
popd