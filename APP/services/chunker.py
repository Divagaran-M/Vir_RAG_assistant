from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages):
    """
    Chunk each page separately while preserving page metadata.

    Input:
    [
        {
            "page": 1,
            "text": "..."
        }
    ]

    Output:
    [
        {
            "text": "...",
            "page": 1
        },
        ...
    ]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"]
                }
            )

    return all_chunks