// Document-wide styling rules. Call via: #show: setup
#let setup(body) = {
  set text(font: "New Computer Modern", size: 10pt)
  set page(
    paper: "us-letter",
    margin: (x: 4%, top: 5%, bottom: 4%),
  )
  set math.equation(numbering: "(1)", block: true)
  show math.equation: set block(spacing: 0.65em)
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)
  set heading(numbering: "1.")
  show heading.where(level: 2): set text(10pt)
  set par(justify: true, first-line-indent: 1em)
  body
}

// Renders the title, authors, and abstract block.
#let title-block(
  title: "Paper Title",
  authors: (),
  abstract: none,
  index-terms: (),
) = {
  set document(title: title, author: authors.map(a => a.name))
  v(3pt, weak: true)
  align(center, text(22pt, title))
  v(8.35mm, weak: true)

  for i in range(calc.ceil(authors.len() / 3)) {
    let end = calc.min((i + 1) * 3, authors.len())
    let is-last = authors.len() == end
    let slice = authors.slice(i * 3, end)
    grid(
      columns: slice.len() * (1fr,),
      gutter: 12pt,
      ..slice.map(author => align(center, {
        text(12pt, author.name)
        if "department" in author [ \ #emph(author.department) ]
        if "organization" in author [ \ #emph(author.organization) ]
        if "location" in author [ \ #author.location ]
        if "email" in author [ \ #link("mailto:" + author.email) ]
      }))
    )
    if not is-last { v(16pt, weak: true) }
  }

  align(center, text(12pt, emph("Electrical and Computer Engineering\nUniversity of Illinois Urbana-Champaign")))
  v(40pt, weak: true)

  if abstract != none [
    #set text(weight: 700)
    #h(1em) _Abstract_---#abstract
  ]
  if index-terms != () [
    #set text(weight: 700)
    #h(1em)_Index terms_---#index-terms.join(", ")
  ]
  v(2pt)
}

// Renders the bibliography.
#let doc-bib(file) = {
  show bibliography: set text(8pt)
  bibliography(file, title: text(10pt)[References], style: "ieee")
}
