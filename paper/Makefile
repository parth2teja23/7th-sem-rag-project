# Build the paper with Tectonic (brew install tectonic). Tectonic runs BibTeX
# and reruns LaTeX as often as needed, so one call is enough.
# Fallback: upload this folder to Overleaf (pdfLaTeX + BibTeX) and compile main.tex.

.PHONY: all pdf clean check

all: pdf

pdf: main.pdf

main.pdf: main.tex numbers.tex refs.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*)
	tectonic --keep-logs main.tex

# Style check from the Task 20 brief; prints nothing when clean.
check:
	@! grep -niE "delve|landscape|seamless|crucial role|worth noting|rapidly evolving|leverag" sections/*.tex
	@! grep -n "Warning: Citation" main.log 2>/dev/null

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
