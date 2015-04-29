package dd.crawler.codeforce.statusparser;

import java.net.URL;

import org.htmlparser.Parser;
import org.htmlparser.Remark;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.visitors.NodeVisitor;



public class SubmissionParser {
	private boolean code_found;
	private String code;

	private NodeVisitor myVisitor = new NodeVisitor() {

		/** 重载抽象类NodeVisitor的beginParsing方法,解析开始时调用此方法 */
		public void beginParsing() {
			//System.out.println("开始解析HTML内容......");
		}

		/** 重载抽象类NodeVisitor的finishedParsing方法,解析结束时调用此方法 */
		public void finishedParsing() {
			//System.out.println("整个HTML内容解析完毕!");
		}

		/** 重载抽象类NodeVisitor的visitTag方法,遇到开始标签时调用此方法 */
		public void visitTag(Tag tag) {
			//System.out.println("开始当前标签: " + tag.getText());
			//if(tag.getText().equals(arg0))
			
			if (tag.getText().indexOf("class=\"prettyprint\"") != -1) {
				code_found = true;
			}
		}

		/** 重载抽象类NodeVisitor的visitEndTag方法,遇到结束标签时调用此方法 */
		public void visitEndTag(Tag tag) {
			//System.out.println("结束当前标签: " + tag.getText());
		}

		/** 重载抽象类NodeVisitor的visitStringNode方法,遇到文本节点时调用此方法 */
		public void visitStringNode(Text string) {
			//System.out.println("当前文本节点: " + string.toPlainTextString());
			if(code_found) {
				code = string.toPlainTextString();
				code_found = false;
			}
			//StringBuilder sb = new StringBuilder(string);
			
		}

		/** 重载抽象类NodeVisitor的visitRemarkNode方法,遇到注释时调用此方法 */
		public void visitRemarkNode(Remark remark) {
			//System.out.println("当前注释: " + remark);
		}
	};
	
	
	public String parseCodeFromUrl(String urlString) throws Exception {
		code_found = false;
		URL url = new URL(urlString);
		Parser parser = new Parser(url.openConnection());
		parser.visitAllNodesWith(myVisitor);
		return code;
	}
	
	public String parseCodeFromHtml(String html) throws Exception {
		Parser parser = new Parser(html);
		code_found = false;
		parser.visitAllNodesWith(myVisitor);
		return code;
	}
}
