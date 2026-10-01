//go:build ignore

// Command check-docs checks Go declaration comments across all build variants.
// Run it from the repository root with go run ./scripts/check-docs.go.
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// main parses source rather than packages so inactive platform and experiment
// files are checked without their toolchains, native libraries, or model assets.
func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	set := token.NewFileSet()
	functions, failures := 0, 0
	problem := func(pos token.Pos, message string) {
		fmt.Fprintf(os.Stderr, "%s: %s\n", set.Position(pos), message)
		failures++
	}
	check := func(name string, pos token.Pos, doc *ast.CommentGroup) {
		if doc == nil || !strings.HasPrefix(doc.Text(), name+" ") {
			problem(pos, "doc comment must begin with "+name+" followed by a description")
		}
	}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if path != root && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "vendor") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		file, err := parser.ParseFile(set, path, nil, parser.ParseComments)
		if err != nil {
			return err
		}
		for _, declaration := range file.Decls {
			switch d := declaration.(type) {
			case *ast.FuncDecl:
				functions++
				check(d.Name.Name, d.Pos(), d.Doc)
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					t, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}
					if t.Name.IsExported() {
						doc := t.Doc
						if doc == nil {
							doc = d.Doc
						}
						check(t.Name.Name, t.Pos(), doc)
					}
					switch body := t.Type.(type) {
					case *ast.InterfaceType:
						for _, method := range body.Methods.List {
							for _, name := range method.Names {
								check(name.Name, name.Pos(), method.Doc)
							}
						}
					case *ast.StructType:
						if t.Name.IsExported() {
							for _, field := range body.Fields.List {
								for _, name := range field.Names {
									if name.IsExported() && field.Doc == nil && field.Comment == nil {
										problem(name.Pos(), t.Name.Name+"."+name.Name+" needs a field comment")
									}
								}
							}
						}
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if failures != 0 {
		fmt.Fprintf(os.Stderr, "%d documentation problems\n", failures)
		os.Exit(1)
	}
	fmt.Printf("Documented %d functions/methods; public types, fields, and interface methods checked across all Go build variants.\n", functions)
}
