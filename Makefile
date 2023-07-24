# source with wildcard
SOURCES := $(wildcard markdown/*.md)
NBS := $(patsubst markdown/%.md, notebooks/%.ipynb, $(SOURCES))

# Set the shell used by make to bash
SHELL := /bin/bash

# rule to run
notebooks/%.ipynb: markdown/%.md
	pandoc --resource-path=assets/ --embed-resources --standalone --wrap=none  $< -o $@
	
	if grep -q 'attachment:assets/[^)]*' $@; then \
	grep -o 'attachment:assets/[^)]*' $@ | while read -r attachment; do \
	v1=$${attachment#attachment:}; \
	v2=$$(grep -A2 "\"$$v1\"" $@ | grep "image/png" | cut -d'"' -f4); \
	v3=$$(sed 's/\\n//g' <<< $$v2); \
	echo "s|$$attachment|data:image/png;base64,$$v3|g" >> sed_commands.txt; \
	done; \
	sed -i -f sed_commands.txt $@; \
	rm sed_commands.txt; \
	fi

all: $(NBS)

clean: 
	rm -f $(NBS)

