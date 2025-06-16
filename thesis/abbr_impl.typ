#let space-default = sym.space.nobreak.narrow
#let style-default(short) = {
	let val = if text.weight <= "medium" { 15% } else { 30% }
	set text(fill: black.lighten(val))
	show: strong
	short
}

#let abbr = state("abbr", (:))
#let abbr-first = state("abbr-first", ())
#let abbr-list = state("abbr-list", ())
#let abbr-space = state("abbr-space", space-default)
#let abbr-style = state("abbr-style", style-default)

#let stringify(text) = {
	if type(text) == str {
		return text
	}
	if "text" in text.fields() {
		return text.text
	}
	panic("Cannot stringify")
}

#let get(short) = {
	let key = stringify(short)
	let entry = abbr.get().at(key, default:none)
	return (key, entry)
}

#let warn(short) = [*(?? #short ??)*]

#let mark-first(key) = {
	if not key in abbr-first.get() { return }
	abbr-first.update(lst => {
		return lst.filter(el => el != key)
	})
}

#let mark-used(key) = {
	if key in abbr-list.get() { return }
	abbr-list.update(lst => {
		lst.push(key)
		return lst
	})
}

/// add single entry
#let add(short, ..entry) = context {
	let long = entry.at(0)
	let plural = entry.at(1, default:none)
	let (key, entry) = get(short)
	if entry == none { // do not update blindly
		let item = (
			l: long,
			pl: plural,
			lbl: label(short + sym.hyph.point + long),
		)
		abbr.update(dct => {
			dct.insert(short, item)
			return dct
		})
		abbr-first.update(lst => {
			lst.push(key)
			return lst
		})
	}
}

/// add list of entries
#let make(..lst) = for (..abbr) in lst.pos() { add(..abbr) }

/// add abbreviations from csv file
#let load(..filename, delimiter:",") = {
	let notempty(s) = { return s != "" }
	let entries = read(..filename).split("\n")
		.filter(notempty)
		.map(entry => csv(bytes(entry),delimiter:delimiter)
			.flatten().map(str.trim).filter(notempty)
		)
	make(..entries)
}

/// short form of abbreviation with link
#let s(short) = context {
	let (key, dct) = get(short)
	if dct == none { return warn(short) }
	mark-used(key)
	let styleit = abbr-style.get()
	if query(dct.lbl).len() != 0 {
		link(dct.lbl, styleit(key))
	} else {
		styleit(key)
	}
}

/// long form of abbreviation
#let l(short) = context {
	let (key, dct) = get(short)
	if dct == none { return warn(short) }
	mark-first(key)
	dct.l
	abbr-space.get()
	sym.paren.l
	sym.zwj
	s(key)
	sym.zwj
	sym.paren.r
}

/// long form _only_
#let lo(short) = context {
	let dct = get(short).at(1)
	if dct == none { return warn(short) }
	dct.l
}

/// automatic short/long form
#let a(short) = context {
	let (key, dct) = get(short)
	if dct == none { return warn(short); }
	if key in abbr-first.get() { l(key) }
	else { s(key) }
}

/// short form plural
#let pls(short) = context {
	let styleit = abbr-style.get()
	[#s(short)#styleit[s]]
}

/// long form plural
#let pll(short) = context {
	let (key, dct) = get(short)
	if dct == none { return warn(short) }
	mark-first(key)
	if dct.pl != none {
		dct.pl
	} else {
		[#dct.l\s]
	}
	abbr-space.get()
	sym.paren.l
	sym.zwj
	pls(key)
	sym.zwj
	sym.paren.r
}

/// long form plural _only_
#let pllo(short) = context {
	let dct = get(short).at(1)
	if dct == none { return warn(short) }
	if dct.pl != none {
		dct.pl
	} else {
		[#dct.l\s]
	}
}

/// automatic short/long form plural
#let pla(short) = context {
	let (key, dct) = get(short)
	// repr(dct)
	// repr(abbr-first.get())
	if dct == none { return warn(short); }
	if key in abbr-first.get() { pll(key) }
	else { pls(key) }
}

/// create list of abbreviations
#let list(
	title: [List of Abbreviations],
	columns: 2,
) = context {
	let lst = abbr.final().pairs()
		.filter(it => it.at(0) in abbr-list.final())
		.map(pair => (s: pair.at(0), l: pair.at(1).l, lbl: pair.at(1).lbl))
		.sorted(key: it => it.s)
	if lst.len() == 0 { return }
	let styleit = abbr-style.get()
	let make-entry(e) = (styleit[#e.s #e.lbl], e.l)
	if columns == 2 {
		let n = int(lst.len()/2)
		let last = if calc.odd(lst.len()) {lst.remove(n)}
		lst = lst.slice(0, n).zip(lst.slice(n)).flatten()
		if last != none { lst.push(last) }
	} else if columns != 1 {
		panic("abbr.list only supports 1 or 2 columns")
	}
	// heading(numbering: none, depth: 3, level: 3, title)
	table(
		columns: (auto, 1fr)*columns,
		stroke:none,
		..for entry in lst { make-entry(entry) }
	)
}

/// configure styling of abbreviations
#let config(style: auto, space-char: auto) = {
	abbr-style.update(it => {
		if style == auto {
			return style-default
		}
		return style
	})
	abbr-space.update(it => {
		if space-char == auto {
			return space-default
		}
		return space-char
	})
}