#!/usr/bin/env python3
"""Solución Ejercicio 1: Encadenamiento hacia adelante y hacia atrás en la biblioteca."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PATRON_ATOMO = re.compile(r"^([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s*$")


@dataclass(frozen=True)
class Atom:
    predicate: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"


def es_variable(termino: str) -> bool:
    return bool(termino) and termino[0].islower()


def parse_atom(texto: str) -> Atom:
    texto = texto.strip()
    m = _PATRON_ATOMO.match(texto)
    if not m:
        raise ValueError(f"No se pudo interpretar el átomo: {texto!r}")
    predicado, args_crudos = m.group(1), m.group(2)
    if args_crudos is None or args_crudos.strip() == "":
        return Atom(predicado, ())
    args = tuple(p.strip() for p in args_crudos.split(","))
    return Atom(predicado, args)


def aplicar_termino(termino: str, subst: dict[str, str]) -> str:
    vistos: set[str] = set()
    while es_variable(termino) and termino in subst and termino not in vistos:
        vistos.add(termino)
        termino = subst[termino]
    return termino


def apply_atom(atomo: Atom, subst: dict[str, str]) -> Atom:
    return Atom(atomo.predicate, tuple(aplicar_termino(a, subst) for a in atomo.args))


def unify(
    a: Atom | str, b: Atom | str, subst: dict[str, str] | None = None
) -> dict[str, str] | None:
    subst = {} if subst is None else dict(subst)

    if isinstance(a, Atom) and isinstance(b, Atom):
        if a.predicate != b.predicate or len(a.args) != len(b.args):
            return None
        for izq, der in zip(a.args, b.args):
            subst = unify(izq, der, subst)
            if subst is None:
                return None
        return subst

    if isinstance(a, Atom) or isinstance(b, Atom):
        return None

    a = aplicar_termino(a, subst)
    b = aplicar_termino(b, subst)
    if a == b:
        return subst
    if es_variable(a):
        if a == aplicar_termino(b, subst):
            return None
        subst[a] = b
        return subst
    if es_variable(b):
        if b == aplicar_termino(a, subst):
            return None
        subst[b] = a
        return subst
    return None


def format_subst(subst: dict[str, str]) -> str:
    if not subst:
        return "{}"
    partes = ", ".join(f"{k}/{v}" for k, v in sorted(subst.items()))
    return "{" + partes + "}"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    premises: tuple[Atom, ...]
    head: Atom

    def __str__(self) -> str:
        cuerpo = " ∧ ".join(str(p) for p in self.premises) if self.premises else "true"
        return f"{self.rule_id}:  {cuerpo}  →  {self.head}"


@dataclass
class KnowledgeBase:
    facts: list[Atom]
    rules: list[Rule]
    query: Atom | None
    source: Path | None = None

    def copy(self) -> KnowledgeBase:
        return KnowledgeBase(list(self.facts), list(self.rules), self.query, self.source)


def load_kb(ruta: str | Path) -> KnowledgeBase:
    ruta = Path(ruta)
    with ruta.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    hechos = [parse_atom(item) for item in (raw.get("facts") or [])]
    reglas = []
    for i, item in enumerate(raw.get("rules") or [], start=1):
        r_id = str(item.get("id") or f"R{i}")
        premisas = tuple(parse_atom(p) for p in (item.get("if") or []))
        cabeza = parse_atom(item["then"])
        reglas.append(Rule(r_id, premisas, cabeza))
    q_raw = raw.get("query")
    consulta = parse_atom(q_raw) if q_raw else None
    return KnowledgeBase(hechos, reglas, consulta, ruta)


def format_kb(kb: KnowledgeBase) -> str:
    lineas = []
    if kb.source:
        lineas.append(f"File: {kb.source}")
    lineas.append("Facts:")
    for f in kb.facts:
        lineas.append(f"  {f}")
    lineas.append("Rules:")
    for r in kb.rules:
        lineas.append(f"  {r}")
    lineas.append(f"Query: {kb.query if kb.query else '(none)'}")
    return "\n".join(lineas)


@dataclass
class TraceEvent:
    message: str


@dataclass
class InferenceResult:
    proved: bool
    subst: dict[str, str] | None
    inferred: list[Atom]
    events: list[TraceEvent] = field(default_factory=list)


def _standardize(rule: Rule, counter: int) -> Rule:
    mapping: dict[str, str] = {}

    def rename_atom(atom: Atom) -> Atom:
        new_args = []
        for arg in atom.args:
            if es_variable(arg):
                mapping.setdefault(arg, f"{arg}_{counter}")
                new_args.append(mapping[arg])
            else:
                new_args.append(arg)
        return Atom(atom.predicate, tuple(new_args))

    return Rule(
        rule.rule_id,
        tuple(rename_atom(p) for p in rule.premises),
        rename_atom(rule.head),
    )


def _satisfy(
    premises: tuple[Atom, ...], facts: list[Atom], subst: dict[str, str]
) -> list[dict[str, str]]:
    if not premises:
        return [subst]
    first, rest = premises[0], premises[1:]
    goal = apply_atom(first, subst)
    results: list[dict[str, str]] = []
    for fact in facts:
        theta = unify(goal, fact, subst)
        if theta is not None:
            results.extend(_satisfy(rest, facts, theta))
    return results


def forward_chain(kb: KnowledgeBase, query: Atom, *, max_rounds: int = 50) -> InferenceResult:
    facts = list(kb.facts)
    inferred: list[Atom] = []
    events = [
        TraceEvent(f"Agenda (given facts): {', '.join(str(f) for f in facts) or '(empty)'}"),
        TraceEvent(f"Query: {query}"),
    ]

    counter = 0
    for round_no in range(1, max_rounds + 1):
        new_atoms: list[Atom] = []
        events.append(TraceEvent(f"--- round {round_no} ---"))
        for rule in kb.rules:
            counter += 1
            fresh = _standardize(rule, counter)
            for theta in _satisfy(fresh.premises, facts, {}):
                head = apply_atom(fresh.head, theta)
                if head in facts or head in new_atoms:
                    continue
                new_atoms.append(head)
                inferred.append(head)
                events.append(TraceEvent(f"Fire {rule.rule_id}  {format_subst(theta)}  ⇒  {head}"))
        if not new_atoms:
            events.append(TraceEvent("No new facts."))
            break
        facts.extend(new_atoms)

    for fact in facts:
        theta = unify(query, fact, {})
        if theta is not None:
            events.append(TraceEvent(f"Query proved: {query}"))
            return InferenceResult(True, theta, inferred, events)

    events.append(TraceEvent("Query not proved."))
    return InferenceResult(False, None, inferred, events)


def backward_chain(kb: KnowledgeBase, query: Atom, *, max_depth: int = 25) -> InferenceResult:
    events = [TraceEvent(f"Query: {query}")]
    counter = [0]
    visited_goals: list[Atom] = []

    def indent(depth: int) -> str:
        return "  " * depth

    def bc_or(
        goal: Atom, subst: dict[str, str], depth: int, stack: tuple[str, ...]
    ) -> dict[str, str] | None:
        goal = apply_atom(goal, subst)
        key = str(goal)
        if key in stack:
            events.append(TraceEvent(f"{indent(depth)}loop: {goal} already on the goal stack — skip"))
            return None
        if depth > max_depth:
            events.append(TraceEvent(f"{indent(depth)}max depth at {goal}"))
            return None

        events.append(TraceEvent(f"{indent(depth)}goal  {goal}"))
        visited_goals.append(goal)

        for fact in kb.facts:
            theta = unify(goal, fact, subst)
            if theta is not None:
                events.append(TraceEvent(f"{indent(depth)}fact   {fact}  {format_subst(theta)}"))
                return theta

        for rule in kb.rules:
            counter[0] += 1
            fresh = _standardize(rule, counter[0])
            theta = unify(goal, fresh.head, subst)
            if theta is None:
                continue
            events.append(TraceEvent(f"{indent(depth)}rule   {rule.rule_id}  {format_subst(theta)}"))
            answer = bc_and(fresh.premises, theta, depth + 1, stack + (key,))
            if answer is not None:
                return answer

        events.append(TraceEvent(f"{indent(depth)}fail   {goal}"))
        return None

    def bc_and(
        premises: tuple[Atom, ...], subst: dict[str, str], depth: int, stack: tuple[str, ...]
    ) -> dict[str, str] | None:
        if not premises:
            return subst
        first, rest = premises[0], premises[1:]
        theta = bc_or(first, subst, depth, stack)
        if theta is None:
            return None
        return bc_and(rest, theta, depth, stack)

    found = bc_or(query, {}, 0, ())
    proved = found is not None
    events.append(TraceEvent("Query proved." if proved else "Query not proved."))

    unique_goals = []
    seen = set()
    for g in visited_goals:
        if str(g) not in seen:
            seen.add(str(g))
            unique_goals.append(g)
    return InferenceResult(proved, found, unique_goals, events)


def correr_consultas(kb: KnowledgeBase, consultas: list[str]) -> None:
    print(format_kb(kb))
    print()

    for q_str in consultas:
        query = parse_atom(q_str)
        fc = forward_chain(kb, query)
        bc = backward_chain(kb, query)

        print("=" * 65)
        print(f"CONSULTA: {query}")
        print("=" * 65)
        print("Forward chaining")
        print("-" * 50)
        for event in fc.events:
            print(event.message)
        print()
        print("Backward chaining")
        print("-" * 50)
        for event in bc.events:
            print(event.message)
        print()
        print("Contrast")
        print("-" * 50)
        print(f"Query {query}:  FC={'YES' if fc.proved else 'NO'}  BC={'YES' if bc.proved else 'NO'}")
        print("Facts FC inferred:  " + (", ".join(str(a) for a in fc.inferred) or "(none)"))
        print("Goals BC visited:   " + (", ".join(str(a) for a in bc.inferred) or "(none)"))
        extra = [a for a in fc.inferred if str(a) not in {str(g) for g in bc.inferred} and a != query]
        if extra:
            print("FC extra work (not needed by BC):  " + ", ".join(str(a) for a in extra))
        print()


def main() -> None:
    root = Path(__file__).resolve().parent
    ruta_yaml = root / "kb" / "biblioteca.yaml"
    if not ruta_yaml.exists():
        ruta_yaml = root / "biblioteca.yaml"

    kb = load_kb(ruta_yaml)
    consultas_obligatorias = [
        "AccesoNoche(Ana)",
        "AccesoNoche(Luis)",
        "AccesoNoche(Marta)",
    ]
    correr_consultas(kb, consultas_obligatorias)

    print("#" * 65)
    print("EXPERIMENTO REPORTE: Añadiendo el hecho TieneCredencial(Luis)")
    print("#" * 65)
    kb_luis = kb.copy()
    kb_luis.facts.append(parse_atom("TieneCredencial(Luis)"))
    correr_consultas(kb_luis, ["AccesoNoche(Luis)"])


if __name__ == "__main__":
    main()