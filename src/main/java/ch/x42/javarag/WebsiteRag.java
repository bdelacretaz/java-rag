package ch.x42.javarag;

//JAVA 17
//DEPS dev.langchain4j:langchain4j:0.31.0
//DEPS dev.langchain4j:langchain4j-open-ai:0.31.0
//DEPS dev.langchain4j:langchain4j-embeddings-bge-small-en-v15-q:0.31.0

// Start this with jbang <filename>
// Based on https://github.com/langchain4j/langchain4j-examples/tree/main/rag-examples

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.document.transformer.HtmlTextExtractor;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.en.v15.BgeSmallEnV15QuantizedEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;
import java.util.Scanner;

public class WebsiteRag {

  static final String OPENAI_API_KEY = "demo";

  static interface Assistant {
    String answer(String query);
  };

  public static void startConversationWith(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("--> ");
                String userQuery = scanner.nextLine();

                if ("exit".equalsIgnoreCase(userQuery)) {
                    break;
                }

                System.out.println(assistant.answer(userQuery));
            }
        }
    }

  /**
   * This example demonstrates how to implement a naive Retrieval-Augmented
   * Generation (RAG) application.
   * By "naive", we mean that we won't use any advanced RAG techniques.
   * In each interaction with the Large Language Model (LLM), we will:
   * 1. Take the user's query as-is.
   * 2. Embed it using an embedding model.
   * 3. Use the query's embedding to search an embedding store (containing small
   * segments of your documents)
   * for the X most relevant segments.
   * 4. Append the found segments to the user's query.
   * 5. Send the combined input (user query + segments) to the LLM.
   * 6. Hope that:
   * - The user's query is well-formulated and contains all necessary details for
   * retrieval.
   * - The found segments are relevant to the user's query.
   */

  public static void main(String[] args) {
    System.out.println("Setting up langchain4j...");
    final Assistant assistant = createAssistant();
    System.out.println("*** RAG engine ready, enter your questions at the --> prompt.");
    startConversationWith(assistant);
  }

  private static Assistant createAssistant() {

    ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
        .apiKey(OPENAI_API_KEY)
        .modelName("gpt-3.5-turbo")
        .build();

    DocumentParser documentParser = new TextDocumentParser();
    HtmlTextExtractor htmlTextExtractor = new HtmlTextExtractor();
    DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
    EmbeddingModel embeddingModel = new BgeSmallEnV15QuantizedEmbeddingModel();
    EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

    final String[] urls = {
        "https://grep.codeconsult.ch/",
        "https://grep.codeconsult.ch/2023/03/24/teaching-programming-sonic-pi-ftw/",
        "https://grep.codeconsult.ch/2023/05/02/from-one-devoxx-to-the-next/",
        "https://grep.codeconsult.ch/2023/03/29/vanilla-js-web-platform/",
        "https://grep.codeconsult.ch/2023/03/28/any-custom-elements-here/",
        "https://grep.codeconsult.ch/2023/03/24/teaching-programming-sonic-pi-ftw/",
        "https://grep.codeconsult.ch/2023/03/16/wwsw-well-written-subtly-wrong/",
        "https://grep.codeconsult.ch/2021/02/04/ssl-tls-certificates-with-lets-encrypt/",
        "https://grep.codeconsult.ch/2020/08/06/how-to-record-decent-conference-videos-without-breaking-the-bank/",
        "https://grep.codeconsult.ch/2018/01/03/great-software-is-like-a-great-music-teacher/",
        "https://grep.codeconsult.ch/2017/12/14/open-source-is-done-welcome-to-open-development/",
        "https://grep.codeconsult.ch/2017/11/23/status-meetings-are-a-waste-of-time-and-money/",
        "https://afkazoo.ch/"
    };
    System.out.println(String.format("Loading %d URLs...", urls.length));
    for (String url : urls) {
      System.err.println("Loading " + url);
      final Document textDocument = UrlDocumentLoader.load(url, documentParser);
      final Document document = htmlTextExtractor.transform(textDocument);
      final List<TextSegment> segments = splitter.split(document);
      final List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
      embeddingStore.addAll(embeddings, segments);
    }

    ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
        .embeddingStore(embeddingStore)
        .embeddingModel(embeddingModel)
        .maxResults(2) // on each interaction we will retrieve the 2 most relevant segments
        .minScore(0.5) // we want to retrieve segments at least somewhat similar to user query
        .build();

    // This must be low if using the openAI demo key, to avoid 
    // getting over the number of tokens limit
    final int maxMessages = 5;
    ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(maxMessages);

    return AiServices.builder(Assistant.class)
        .chatLanguageModel(chatLanguageModel)
        .contentRetriever(contentRetriever)
        .chatMemory(chatMemory)
        .build();
  }
}