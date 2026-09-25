from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


def api_request(method: str, path: str, **kwargs: Any) -> httpx.Response:
	"""Call the FastAPI service and turn transport errors into UI-friendly errors."""
	base_url = st.session_state.api_url.rstrip("/")
	try:
		with httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT) as client:
			return client.request(method, path, **kwargs)
	except httpx.RequestError as exc:
		raise RuntimeError(
			f"Could not connect to the RAG API at {base_url}. Start the API and try again."
		) from exc


def response_error(response: httpx.Response) -> str:
	try:
		detail = response.json().get("detail")
	except (ValueError, TypeError):
		detail = None
	return str(detail or f"Request failed with status {response.status_code}.")


def render_health() -> None:
	try:
		response = api_request("GET", "/health")
		if response.is_success:
				health = response.json()
				st.success(f"API online | {health.get('environment', 'unknown')} environment")
		else:
			st.error(f"API returned {response.status_code}: {response_error(response)}")
	except RuntimeError as exc:
		st.error(str(exc))


def render_sources(payload: dict[str, Any]) -> None:
	sources = payload.get("retrieved_sources", [])
	citations = payload.get("citations", [])
	if citations:
		st.markdown("#### Citations")
		for citation in citations:
			location = []
			if citation.get("page_number") is not None:
				location.append(f"page {citation['page_number']}")
			if citation.get("section"):
				location.append(citation["section"])
			suffix = f" · {' · '.join(location)}" if location else ""
			st.markdown(f"- **{citation['filename']}**{suffix}")

	if sources:
		with st.expander(f"Retrieved context ({len(sources)} chunks)"):
			for index, source in enumerate(sources, start=1):
				metadata = [source["filename"], f"score {source['score']:.3f}"]
				if source.get("page_number") is not None:
					metadata.append(f"page {source['page_number']}")
				if source.get("section"):
					metadata.append(source["section"])
				st.markdown(f"**{index}. {' | '.join(metadata)}**")
				st.caption(source["text"])


def ingest_document(uploaded_file: Any) -> None:
	try:
		response = api_request(
			"POST",
			"/documents",
			files={
				"file": (
					uploaded_file.name,
					uploaded_file.getvalue(),
					uploaded_file.type or "application/octet-stream",
				)
			},
		)
	except RuntimeError as exc:
		st.error(str(exc))
		return

	if not response.is_success:
		st.error(response_error(response))
		return

	document = response.json()
	st.session_state.documents[document["document_id"]] = document
	st.success(
		f"Indexed {document['filename']} · {document['chunks_indexed']} chunks"
	)


def render_sidebar() -> None:
	with st.sidebar:
		st.markdown("## RAG control room")
		st.caption("Index documentation, then interrogate it with grounded answers.")

		st.text_input("API URL", key="api_url", help="The FastAPI service URL.")
		st.markdown("### Service")
		render_health()

		st.markdown("### Add documentation")
		uploaded_file = st.file_uploader(
			"PDF or Markdown file",
			type=["pdf", "md", "markdown"],
			help="Files are sent to the API and indexed in Qdrant.",
		)
		if uploaded_file is not None:
			if st.button("Index document", type="primary", use_container_width=True):
				with st.spinner("Parsing and indexing document..."):
					ingest_document(uploaded_file)

		if st.session_state.documents:
			st.markdown("### Indexed this session")
			for document_id, document in list(st.session_state.documents.items()):
				columns = st.columns([4, 1])
				columns[0].caption(
					f"{document['filename']} | {document['chunks_indexed']} chunks"
				)
				if columns[1].button("×", key=f"delete-{document_id}", help="Delete document"):
					try:
						response = api_request("DELETE", f"/documents/{document_id}")
						if response.is_success:
							del st.session_state.documents[document_id]
							st.rerun()
						st.error(response_error(response))
					except RuntimeError as exc:
						st.error(str(exc))


def main() -> None:
	st.set_page_config(
		page_title="Axentra RAG Workbench",
			page_icon="R",
		layout="wide",
		initial_sidebar_state="expanded",
	)

	if "api_url" not in st.session_state:
		st.session_state.api_url = os.getenv("RAG_API_URL", DEFAULT_API_URL)
	if "documents" not in st.session_state:
		st.session_state.documents = {}

	st.markdown(
		"""
		<style>
		:root { --ink: #102a43; --muted: #627d98; --accent: #d64545; }
		.stApp { background: #f7f9fb; }
		[data-testid="stSidebar"] { background: #102a43; }
		[data-testid="stSidebar"] * { color: #f7f9fb; }
		[data-testid="stSidebar"] input { color: #102a43; }
		.hero { padding: 1.25rem 0 2rem; border-bottom: 1px solid #d9e2ec; }
		.hero-kicker { color: var(--accent); font-size: .75rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
		.hero h1 { color: var(--ink); font-size: clamp(2rem, 5vw, 4.6rem); line-height: .98; margin: .35rem 0 .8rem; }
		.hero p { color: var(--muted); font-size: 1.05rem; max-width: 42rem; }
		[data-testid="stMetricValue"] { color: var(--ink); }
		</style>
		<div class="hero">
			<div class="hero-kicker">Axentra · retrieval augmented generation</div>
			<h1>Ask your documentation<br>better questions.</h1>
			<p>Upload the source material, then get answers that stay tethered to the indexed evidence.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	render_sidebar()

	query_col, info_col = st.columns([1.6, 1], gap="large")
	with query_col:
		st.markdown("### Query the knowledge base")
		with st.form("query-form"):
			question = st.text_area(
				"Question",
				placeholder="How do I authenticate requests?",
				height=130,
				label_visibility="collapsed",
			)
			controls = st.columns([1, 1, 2])
			with controls[0]:
				llm_enabled = st.toggle("Generate answer", value=True)
			with controls[2]:
				submitted = st.form_submit_button(
					"Search documentation", type="primary", use_container_width=True
				)

		if submitted:
			if not question.strip():
				st.warning("Enter a question first.")
			else:
				try:
					with st.spinner("Retrieving evidence..."):
						response = api_request(
							"POST",
							"/query",
							json={"question": question.strip(), "llm_enabled": llm_enabled},
						)
					if response.is_success:
						st.session_state.last_query = response.json()
					else:
						st.error(response_error(response))
				except RuntimeError as exc:
					st.error(str(exc))

		result = st.session_state.get("last_query")
		if result:
			status = "Grounded answer" if result.get("grounded") else "Insufficient evidence"
			if result.get("cached"):
				status += " · cached"
			st.markdown(f"#### {status}")
			if result.get("llm_enabled") and result.get("answer"):
				st.info(result["answer"])
			else:
				st.caption("Answer generation is off. Review the retrieved context below.")
			render_sources(result)

	with info_col:
		st.markdown("### Pipeline signals")
		result = st.session_state.get("last_query")
		if result:
			metrics = st.columns(2)
			metrics[0].metric("Sources", len(result.get("retrieved_sources", [])))
			metrics[1].metric("Citations", len(result.get("citations", [])))
			st.caption("Evidence is retrieved, validated, and optionally passed to the configured LLM.")
		else:
			st.caption("Your answer, grounding status, cache state, and source chunks will appear here after the first query.")


if __name__ == "__main__":
	main()

