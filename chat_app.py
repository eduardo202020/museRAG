from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from app.config import get_settings
from app.rag_service import RagService
from app.schemas import ChatQueryRequest


class MuseRAGDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MuseRAG Chat")
        self.root.geometry("980x720")

        self.settings = get_settings()
        self.service = RagService(self.settings)
        self.is_busy = False

        self.status_var = tk.StringVar()
        self.room_var = tk.StringVar()
        self.artwork_var = tk.StringVar()
        self.top_k_var = tk.StringVar(value=str(self.settings.muserag_top_k))

        self._build_layout()
        self._refresh_status()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        header = ttk.Label(
            container,
            text="Consulta la base vectorial del museo",
            font=("TkDefaultFont", 14, "bold"),
        )
        header.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text="Haz una pregunta y la app respondera usando el contenido indexado en Chroma.",
        )
        subtitle.pack(anchor="w", pady=(4, 12))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Sala").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.room_var, width=18).grid(row=0, column=1, sticky="ew", padx=(0, 12))

        ttk.Label(controls, text="Obra").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.artwork_var, width=18).grid(row=0, column=3, sticky="ew", padx=(0, 12))

        ttk.Label(controls, text="Top K").grid(row=0, column=4, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.top_k_var, width=8).grid(row=0, column=5, sticky="w")

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(container, text="Pregunta").pack(anchor="w")
        self.question_input = scrolledtext.ScrolledText(container, height=5, wrap="word")
        self.question_input.pack(fill="x", pady=(4, 12))

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=(0, 12))

        self.ask_button = ttk.Button(actions, text="Preguntar", command=self._submit_question)
        self.ask_button.pack(side="left")

        self.rebuild_button = ttk.Button(actions, text="Reconstruir indice", command=self._rebuild_index)
        self.rebuild_button.pack(side="left", padx=(8, 0))

        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        ttk.Label(container, text="Respuesta").pack(anchor="w")
        self.answer_output = scrolledtext.ScrolledText(container, height=10, wrap="word", state="disabled")
        self.answer_output.pack(fill="both", expand=False, pady=(4, 12))

        ttk.Label(container, text="Fuentes recuperadas").pack(anchor="w")
        self.sources_output = scrolledtext.ScrolledText(container, height=16, wrap="word", state="disabled")
        self.sources_output.pack(fill="both", expand=True)

    def _set_busy(self, busy: bool, status_text: str | None = None) -> None:
        self.is_busy = busy
        state = "disabled" if busy else "normal"
        self.ask_button.config(state=state)
        self.rebuild_button.config(state=state)
        if status_text:
            self.status_var.set(status_text)

    def _refresh_status(self) -> None:
        try:
            total = self.service.count_documents()
            if total > 0:
                self.status_var.set(f"Base lista: {total} fragmentos indexados")
            else:
                self.status_var.set("Base vacia: usa 'Reconstruir indice'")
        except Exception as exc:
            self.status_var.set(f"Error de estado: {exc}")

    def _submit_question(self) -> None:
        if self.is_busy:
            return

        question = self.question_input.get("1.0", "end").strip()
        if not question:
            messagebox.showwarning("Pregunta requerida", "Escribe una pregunta antes de consultar.")
            return

        try:
            top_k = int(self.top_k_var.get().strip() or self.settings.muserag_top_k)
        except ValueError:
            messagebox.showwarning("Top K invalido", "Top K debe ser un numero entero.")
            return

        payload = ChatQueryRequest(
            question=question,
            room_id=self.room_var.get().strip() or None,
            artwork_id=self.artwork_var.get().strip() or None,
            top_k=top_k,
        )

        self._set_busy(True, "Consultando la base vectorial...")
        worker = threading.Thread(target=self._run_question, args=(payload,), daemon=True)
        worker.start()

    def _run_question(self, payload: ChatQueryRequest) -> None:
        try:
            answer, sources, _meta = self.service.answer_question(payload)
            self.root.after(0, self._show_response, answer, sources)
        except Exception as exc:
            self.root.after(0, self._show_error, "No se pudo completar la consulta", str(exc))

    def _rebuild_index(self) -> None:
        if self.is_busy:
            return

        self._set_busy(True, "Reconstruyendo indice...")
        worker = threading.Thread(target=self._run_rebuild, daemon=True)
        worker.start()

    def _run_rebuild(self) -> None:
        try:
            total = self.service.rebuild_index()
            self.root.after(0, self._on_rebuild_success, total)
        except Exception as exc:
            self.root.after(0, self._show_error, "No se pudo reconstruir el indice", str(exc))

    def _on_rebuild_success(self, total: int) -> None:
        self._set_busy(False)
        self.status_var.set(f"Indice reconstruido: {total} fragmentos cargados")
        messagebox.showinfo("Indice listo", f"Se cargaron {total} fragmentos en la base vectorial.")

    def _show_response(self, answer: str, sources: list) -> None:
        self._write_text(self.answer_output, answer)

        source_blocks: list[str] = []
        for index, source in enumerate(sources, start=1):
            page = source.metadata.get("page")
            extra = f" | pagina={page}" if page else ""
            source_blocks.append(
                f"[{index}] {source.kind} | score={source.score:.3f}{extra}\n"
                f"source={source.source}\n"
                f"{source.text}"
            )

        sources_text = "\n\n" + ("-" * 80) + "\n\n"
        rendered_sources = sources_text.join(source_blocks) if source_blocks else "No se recuperaron fuentes."
        self._write_text(self.sources_output, rendered_sources)

        self._set_busy(False)
        self._refresh_status()

    def _show_error(self, title: str, detail: str) -> None:
        self._set_busy(False)
        self._refresh_status()
        messagebox.showerror(title, detail)

    @staticmethod
    def _write_text(widget: scrolledtext.ScrolledText, content: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = MuseRAGDesktopApp(root)
    root.minsize(840, 620)
    root.mainloop()


if __name__ == "__main__":
    main()
