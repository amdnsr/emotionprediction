<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">os</span>
<span class="kn">from</span> <span class="nn">pprint</span> <span class="kn">import</span> <span class="n">pprint</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">matplotlib</span> <span class="kn">import</span> <span class="n">pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="kn">import</span> <span class="nn">cv2</span>
<span class="kn">from</span> <span class="nn">PIL</span> <span class="kn">import</span> <span class="n">Image</span>
<span class="kn">import</span> <span class="nn">keras</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">from</span> <span class="nn">homomorphicfilter</span> <span class="kn">import</span> <span class="n">HomomorphicFilter</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">homomorphicfilter</span> <span class="o">=</span> <span class="n">HomomorphicFilter</span><span class="p">(</span><span class="n">a</span> <span class="o">=</span> <span class="mf">0.75</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="mf">1.25</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">model_path</span> <span class="o">=</span> <span class="s2">"models/emotion_model.hdf5"</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">keras</span><span class="o">.</span><span class="n">models</span><span class="o">.</span><span class="n">load_model</span><span class="p">(</span><span class="n">model_path</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell-outputWrapper">

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">

<pre>WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">label_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'Angry'</span><span class="p">,</span> <span class="s1">'Disgust'</span><span class="p">,</span> <span class="s1">'Fear'</span><span class="p">,</span> <span class="s1">'Happy'</span><span class="p">,</span> <span class="s1">'Sad'</span><span class="p">,</span> <span class="s1">'Surprise'</span><span class="p">,</span> <span class="s1">'Neutral'</span><span class="p">]</span>
<span class="n">label_indices</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">7</span><span class="p">))</span>
<span class="n">id_name_map</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">label_indices</span><span class="p">,</span> <span class="n">label_names</span><span class="p">))</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">filterd_and_normalized</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">):</span>
    <span class="n">img_filtered</span> <span class="o">=</span> <span class="n">homomorphicfilter</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="n">I</span><span class="o">=</span><span class="n">img</span><span class="p">,</span> <span class="n">filter_params</span><span class="o">=</span><span class="p">[</span><span class="mi">30</span><span class="p">,</span><span class="mi">2</span><span class="p">])</span>
    <span class="n">img_normalized</span> <span class="o">=</span> <span class="n">cv2</span><span class="o">.</span><span class="n">equalizeHist</span><span class="p">(</span><span class="n">img_filtered</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">img_normalized</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">load_image_as_grayscale_and_apply_filter_and_normalization</span><span class="p">(</span><span class="n">img_path_in</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">):</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">Image</span><span class="o">.</span><span class="n">open</span><span class="p">(</span><span class="n">img_path_in</span><span class="p">)</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">img</span><span class="o">.</span><span class="n">convert</span><span class="p">(</span><span class="s2">"L"</span><span class="p">)</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
    <span class="n">img_normalized</span> <span class="o">=</span> <span class="n">filterd_and_normalized</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">img_normalized</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">reshape_for_model</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">dim</span><span class="p">):</span>
    <span class="n">resized</span> <span class="o">=</span> <span class="n">cv2</span><span class="o">.</span><span class="n">resize</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">dim</span><span class="p">,</span> <span class="n">interpolation</span> <span class="o">=</span> <span class="n">cv2</span><span class="o">.</span><span class="n">INTER_AREA</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">resized</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="o">*</span><span class="p">(</span><span class="n">resized</span><span class="o">.</span><span class="n">shape</span><span class="p">),</span> <span class="mi">1</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">predict_image</span><span class="p">(</span><span class="n">img_path_in</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">id_name_map</span><span class="p">,</span> <span class="n">dim</span><span class="p">):</span>
    <span class="n">filteredandnormalized</span> <span class="o">=</span> <span class="n">load_image_as_grayscale_and_apply_filter_and_normalization</span><span class="p">(</span><span class="n">img_path_in</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">)</span>
    <span class="n">model_in</span> <span class="o">=</span> <span class="n">reshape_for_model</span><span class="p">(</span><span class="n">filteredandnormalized</span><span class="p">,</span> <span class="n">dim</span><span class="p">)</span>
    <span class="n">preds</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">model_in</span><span class="p">)</span>
    <span class="n">max_ind</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">preds</span><span class="p">)</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">max_ind</span><span class="p">,</span> <span class="n">id_name_map</span><span class="p">[</span><span class="n">max_ind</span><span class="p">])</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">def</span> <span class="nf">test_model_accuracy</span><span class="p">(</span><span class="n">test_image_folder</span><span class="p">):</span>
    <span class="n">path</span><span class="p">,</span> <span class="n">dirs</span><span class="p">,</span> <span class="n">files</span> <span class="o">=</span> <span class="nb">next</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">walk</span><span class="p">(</span><span class="n">test_image_folder</span><span class="p">))</span>
    <span class="n">results</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="n">files</span><span class="p">),))</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">file</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">files</span><span class="p">):</span>
        <span class="n">img_path_in</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">path</span><span class="p">,</span> <span class="n">file</span><span class="p">)</span>
        <span class="n">max_ind</span><span class="p">,</span> <span class="n">id_name_map</span><span class="p">[</span><span class="n">max_ind</span><span class="p">]</span> <span class="o">=</span> <span class="n">predict_image</span><span class="p">(</span><span class="n">img_path_in</span><span class="p">,</span> <span class="n">homomorphicfilter</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">id_name_map</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="mi">64</span><span class="p">))</span>
        <span class="n">results</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">max_ind</span>
    <span class="k">return</span> <span class="n">results</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">label_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'Angry'</span><span class="p">,</span> <span class="s1">'Disgust'</span><span class="p">,</span> <span class="s1">'Fear'</span><span class="p">,</span> <span class="s1">'Happy'</span><span class="p">,</span> <span class="s1">'Sad'</span><span class="p">,</span> <span class="s1">'Surprise'</span><span class="p">,</span> <span class="s1">'Neutral'</span><span class="p">]</span>
<span class="n">label_indices</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">7</span><span class="p">))</span>
<span class="n">id_name_map</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="nb">zip</span><span class="p">(</span><span class="n">label_indices</span><span class="p">,</span> <span class="n">label_names</span><span class="p">))</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">emotion_index</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">emotion_name</span> <span class="o">=</span> <span class="n">id_name_map</span><span class="p">[</span><span class="n">emotion_index</span><span class="p">]</span>
<span class="n">test_image_folder</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"archive/test/</span><span class="si">{</span><span class="n">emotion_name</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">emotion_name</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell-outputWrapper">

<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

<div class="jp-OutputPrompt jp-OutputArea-prompt">Out[ ]:</div>

<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">

<pre>'Angry'</pre>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">results</span> <span class="o">=</span> <span class="n">test_model_accuracy</span><span class="p">(</span><span class="n">test_image_folder</span><span class="p">)</span>
<span class="n">accuracy</span> <span class="o">=</span> <span class="p">(</span><span class="n">results</span> <span class="o">==</span> <span class="n">emotion_index</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">/</span><span class="p">(</span><span class="n">results</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Accuracy for</span> <span class="si">{</span><span class="n">emotion_name</span><span class="si">}</span> <span class="s2">=</span> <span class="si">{</span><span class="nb">int</span><span class="p">(</span><span class="mi">10000</span><span class="o">*</span><span class="n">accuracy</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span><span class="o">/</span><span class="mf">100.0</span><span class="si">}</span><span class="s2">%"</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell-inputWrapper">

<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">

# Test all labels[¶](#Test-all-labels)

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">accuracy_dict</span> <span class="o">=</span> <span class="p">{}</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">label_name</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">label_names</span><span class="p">):</span>
    <span class="n">test_image_folder</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"archive/test/</span><span class="si">{</span><span class="n">label_name</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span>
    <span class="n">results</span> <span class="o">=</span> <span class="n">test_model_accuracy</span><span class="p">(</span><span class="n">test_image_folder</span><span class="p">)</span>
    <span class="n">accuracy</span> <span class="o">=</span> <span class="p">(</span><span class="n">results</span> <span class="o">==</span> <span class="n">i</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">/</span><span class="p">(</span><span class="n">results</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
    <span class="n">accuracy_dict</span><span class="p">[</span><span class="n">label_name</span><span class="p">]</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mi">10000</span><span class="o">*</span><span class="n">accuracy</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span><span class="o">/</span><span class="mf">100.0</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Completed the label</span> <span class="si">{</span><span class="n">label_name</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">

<div class="jp-Cell-inputWrapper">

<div class="jp-InputArea jp-Cell-inputArea">

<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>

<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">

<div class="CodeMirror cm-s-jupyter">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">pprint</span><span class="p">(</span><span class="n">accuracy_dict</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

</div>