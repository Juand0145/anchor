# Anclas, no párrafos: extraccion de textos apalancados en LLMs de la fuente y no en los modelos.

## El problema

Cuando hay que sacar información etiquetada de un documento —y esa información tiene que ser de confianza— lo habitual es poner un LLM de intermediario. El modelo lee un trozo del documento y devuelve el texto que algunas veces termina siendo etiquetado en algunos caso las salida de un modelo se verìa a algo similar ha esto.

```json
{
  "categoria": "definicion",
  "etiqueta": "160.103",
  "texto": "Breach means the acquisition, access, use, or disclosure of protected health information in a manner not permitted under subpart E of this part which compromises the security or privacy of the protected health information. (1) Breach excludes: (i) Any unintentional acquisition, access, or use of protected health information by a workforce member or person acting under the authority of a covered entity or a business associate, if such acquisition, access, or use was made in good faith and within the scope of authority and does not result in further use or disclosure in a manner not permitted under subpart E of this part. (ii) Any inadvertent disclosure by a person who is authorized to access protected health information at a covered entity or business associate to another person authorized to access protected health information at the same covered entity or business associate, and the information received as a result of such disclosure is not further used or disclosed in a manner not permitted under subpart E of this part. (iii) A disclosure of protected health information where a covered entity or business associate has a good faith belief that an unauthorized person to whom the disclosure was made would not reasonably have been able to retain such information."
}
```

El modelo no solo decide la categoría. También **reescribe o copia el cuerpo entero** de cada unidad. Para que eso funcione hay que aceptar tres restricciones:

1. **Ventanas de contexto cortas.** La salida tiene que caber el texto completo más sus etiquetas. Si la unidad es larga, la ventana de entrada se achica para dejar sitio a la respuesta.
2. **Ventanas solapadas.** Hay que garantizar que todo el texto de una categoría caiga dentro de una sola ventana. El solape vuelve a mandar los mismos tokens para no cortar una unidad a la mitad.
3. **Modelos relativamente potentes.** Copiar cláusulas largas sin parafrasear, sin omitir y sin inventar exige un modelo que aguante bien la instrucción literal.

Eso produce cuatro problemas concretos:

- Se desaprovechan las ventanas de contexto grandes: la mayor parte del presupuesto se gasta en repetir el documento en la salida.
- Se reprocesan tokens. El solape vuelve a pagar el mismo texto en la llamada siguiente.
- El costo sube con la dificultad de la extracción, no solo con el tamaño del documento.
- Una ventana recortada empuja al modelo a cerrar la unidad como pueda. Ahí aparecen las alucinaciones: cláusulas parafraseadas, frases omitidas o texto que suena correcto y no está en la fuente.

## La salida con anclas

El sistema de anclas cambia el contrato. El LLM no busca el texto para reescribirlo. Mira dónde empieza la unidad y deja un ancla de inicio. Mira dónde termina y deja un ancla de fin. La salida se parece a esto:

```json
{
  "categoria": "definicion",
  "etiqueta": "160.103",
  "start_anchor": "Breach means the acquisition, access, use, or disclosure of protected health information",
  "end_anchor": "to whom the disclosure was made would not reasonably have been able to retain such information."
}
```

Es la misma unidad que arriba: la categoría y la etiqueta se quedan, y en lugar del `texto` completo hay dos anclas.

`start_anchor` y `end_anchor` son substrings literales del documento, cortos (unas doce a veinticinco palabras), no el párrafo. El motor localiza esas cadenas dentro del texto fuente y corta el span. El cuerpo de la unidad lo produce el código, no el modelo.

La categoría y la etiqueta se quedan en la salida. El motor no las lee ni las valida: solo las pasa al JSON.

Con eso cambia el presupuesto:

- **Menos tokens de salida.** La respuesta son fronteras, no el documento otra vez.
- **Ventanas de entrada más grandes.** Al no reservar la salida para copiar el cuerpo, cabe más documento por llamada.
- **Sin solape para recuperar texto.** No hace falta reenviar el mismo pasaje para que el modelo lo copie entero. Si una unidad cruza el corte del chunk, el motor guarda una sola unidad pendiente y la cierra en el chunk siguiente, hacia adelante.
- **Modelos más ligeros.** La tarea es señalar fronteras verbatim, no redactar ni sostener un párrafo largo sin alterarlo.
- **El cuerpo no se alucina.** El modelo no escribe el contenido. Si el ancla no está en el documento, o no es única, el motor no inventa el párrafo: deja el segmento sin resolver y lo marca. El texto que sí sale es un corte del fuente, buscado con las anclas.

## Comparación de las dos salidas

Misma unidad, dos contratos. En la extracción normal el modelo escribe el cuerpo completo. Con anclas escribe solo dos fronteras. El `texto` crece con la unidad. Las anclas no.

## Precio

El costo por llamada es `tokens_entrada × precio_entrada + tokens_salida × precio_salida`. Con anclas baja sobre todo la salida, y la entrada no paga solape.

El modelo por defecto del repo es `claude-haiku-4-5-20251001` (`anchor_extract/settings.py`). Su tarifa, según la [página de precios de Anthropic](https://docs.claude.com/en/docs/about-claude/pricing) consultada el 10 de noviembre de 2026, es **$1 por millón de tokens de entrada y $5 por millón de salida**. La salida cuesta cinco veces más que la entrada, y ahí es donde cambia el contrato.

Para el ejemplo de arriba, con la heurística del repo (`len//4 + 1`, no un tokenizador real):

| | JSON normal | JSON con anclas |
|---|---|---|
| Caracteres | 1352 | 281 |
| Tokens estimados de salida | ~339 | ~71 |
| Solo el cuerpo (`texto` vs. las dos anclas) | ~321 | ~46 |

La salida con anclas es unas cinco veces más pequeña para esta unidad. Con una unidad diez veces más larga, el `texto` pasaría a unos 3200 tokens y las anclas seguirían en unos 46.

Con Haiku 4.5, para 1,000 unidades del tamaño del ejemplo (supuesto: todas miden lo mismo):

| | Extracción normal | Con anclas |
|---|---|---|
| Tokens de salida | ~339,000 | ~71,000 |
| Costo de salida ($5 / MTok) | ~$1.70 | ~$0.36 |

La diferencia es de unos $1.34 por cada 1,000 unidades, solo en salida. Es un ahorro de cerca del ochenta por ciento, no de un orden de magnitud, y es pequeño en términos absolutos para un documento corto. Pesa cuando se procesan muchos documentos o unidades largas.

Contra eso, el contrato fijo: `anchor.txt` más el perfil HIPAA suman unos 4,000 tokens por llamada con la misma heurística (sin contar el schema de la herramienta). A $1 / MTok son unos $0.004 por llamada sin caché. Si una llamada cubre varias unidades, ese costo se reparte. Con la entrada en caché, la tarifa de lectura de Haiku 4.5 es $0.10 / MTok. No medí cuántas unidades caben en cada llamada, así que no calculé el total. Este cálculo tampoco cuenta el prompt de la extracción normal, que también existe.

Haiku 5.5 aparece en la misma página con $0.10 / MTok de entrada y $0.50 / MTok de salida para prompts de hasta 100,000 tokens. La proporción entre los dos métodos no cambia, pero el costo absoluto baja.

No es gratis: cada llamada lleva el contrato fijo (con `cache_control`), hay reintentos, las unidades marcadas se revisan a mano, y la pasada opcional `refine_implicit_ends` suma una llamada por unidad marcada.

Para medirlo con tu documento, corre `scripts/sweep_chunk_budget.py` (`--live` para llamadas reales) y compara `n_calls` y los tokens contra la extracción normal sobre el mismo rango.
