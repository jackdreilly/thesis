# Makefile for the kdd paper
# Most of the elements are shamelessly copied from the wikipedia page
# http://en.wikipedia.org/wiki/Make_%28software%29
# To generate the figures, you have to start openoffice or libreoffice with the command:
# ooffice "-accept=socket,host=localhost,port=2002,tcpNoDelay=1;urp;"


PAPER	= presentation.pdf
SRC = presentation.tex
# All the tex files
TEX	= $(wildcard *.tex)
UNOCONV	= unoconv -f pdf 


all: $(PAPER) open

$(PAPER): $(TEX) figures
	pdflatex $(SRC) ;\
	# bibtex article ;\
	# pdflatex $(SRC) ;\
	# pdflatex $(SRC) ;\
	mv presentation.pdf $(PAPER)

clean:
	rm -f *~
	rm -f *pdf
	rm -f *log
	rm -f *aux
	rm -f *bbl
	rm -f *blg

auxclean: clean
	rm -f *fls
	rm -f *fdb_latexmk
	rm -f *aux
	rm -f *bbl
	rm -rf *blg
	rm -rf *synctex.gz
	rm -rf *blg

figures: figs-gen/junctions.pdf figs-gen/junctions-riemann.pdf figs-gen/junctions-riemann-rs.pdf figs-gen/junctions-riemann-rs-one.pdf figs-gen/riemann-junction.pdf figs-gen/shock.pdf figs-gen/god-rp.pdf figs-gen/god-rp-sln.pdf figs-gen/complexity.pdf


figs-gen/%.pdf: figures/%.odg
	echo "Making figure" $* ;\
	${UNOCONV} figures/$*.odg; \
	mv figures/$*.pdf figs-gen/$*.pdf

# works on mac
open:
	open $(PAPER)

allmac: all open

arxiv: $(PAPER)
	zip arxiv.zip *.tex *.cls *.{bbl,blg} 00README.XXX figures/*{pdf,png,pdf_tex} figs-plots/*pdf figs-gen/*pdf TikZ/*.pdf
