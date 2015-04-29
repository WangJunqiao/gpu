package dd.crawler.hdoj;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.SQLException;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;

import dd.crawler.codeforce.db.StatusSql;
import dd.crawler.codeforce.db.VisitedSubmissionSql;
import dd.crawler.codeforce.meta.Submission;
import dd.crawler.codeforce.statusparser.SubmissionParser;

public class CodeForceSubmissionCrawler {

	private static final String LOG_FILE = "e:/submission_log.out";

	public static String getHtml(String url) throws ClientProtocolException, IOException  {
		HttpClient httpClient = new DefaultHttpClient();

		//httpClient.getParams().setIntParameter("http.connection-manager.timeout", 60000);
		HttpGet httpGet = new HttpGet(url);

		HttpResponse response = httpClient.execute(httpGet);
		HttpEntity entity = response.getEntity();
		return EntityUtils.toString(entity);
	}

	private static BlockingQueue<Submission> bQueue = new LinkedBlockingQueue<Submission>();

	class HtmlGetter extends Thread {
		public static final int ThreadNumber = 1;
		public static final int BasePageId = 0;
		public static final int Batch = 10;
		private static final int InitialSleepTime = 20;
		private static final String oo = "I am a lock";

		private int threadId;
		public boolean stop;

		public HtmlGetter(int threadId) {
			this.threadId = threadId;
		}

		@Override
		public void run() {
			System.err.println("threadId = " + threadId + " is running...");
			stop = false;
			int sleep_time = InitialSleepTime;
			for (int loop = 0; !stop; loop++) {

				int start = BasePageId + ThreadNumber * loop * Batch + threadId * Batch;
				List<Submission> list = null;
				try {
					list = StatusSql.getStatus(start, Batch);
				} catch (SQLException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
				
				for (int i=0;i<list.size();i++) {
					Submission sub = list.get(i);
					if(stop) break;
					try {
						if (VisitedSubmissionSql.isVisited(sub.getSubmissionId())) {
							continue;
						}
						System.err.println("thread " + threadId
								+ " sleep for " + sleep_time + " ms");
						sleep(sleep_time);
						//oo.wait(sleep_time);
						sleep_time *= 4;
						sleep_time = Math.min(sleep_time, 7200000); //最少两小时测试一次 
						System.err.println("thread " + threadId
								+ " is downloading submission " + sub.getSubmissionId());
						String url = "http://codeforces.com/contest/"
								+ sub.getContestId() + "/submission/"
								+ sub.getSubmissionId();

						long t = System.currentTimeMillis();
						sub.setCode(getHtml(url));
						bQueue.add(sub);
						System.err.println(sub.getSubmissionId());
						long tot = System.currentTimeMillis() - t;
						System.out.println("submissionId = " + sub.getSubmissionId()
								+ ", threadId = " + threadId + ", time used = " + tot
								+ " ms");
						sleep_time = InitialSleepTime;
						//oo.notifyAll();
					} catch (Exception e) {
						//e.printStackTrace();
						System.err.println("thread " + threadId
								+ " will redownload "+sub.getSubmissionId());
						i--; //retry
					}
				}

			}
			System.err.println("threadId = " + threadId + " is shutdown.");
		}
	}

	private boolean stop_parser;

	private void runCrawler() throws Exception {
		PrintStream out = new PrintStream(new File(LOG_FILE));
		System.setOut(out);

		HtmlGetter htmlGetter[] = new HtmlGetter[HtmlGetter.ThreadNumber];
		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i] = new HtmlGetter(i);
			htmlGetter[i].start();
		}
		stop_parser = false;

		new Thread() {
			@Override
			public void run() {
				Scanner scanner = new Scanner(System.in);
				while (scanner.hasNext()) {
					String line = scanner.nextLine();
					if (line.equals("quit")) {
						stop_parser = true;
						break;
					}
				}
				System.err.println("deamon thread stopped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			}
		}.start();

		SubmissionParser parser = new SubmissionParser();
		while (true) {
			while (bQueue.size() == 0 && !stop_parser) {
				Thread.currentThread().sleep(20);
			}
			if (stop_parser)
				break;
			Submission sub = bQueue.remove();
			long t = System.currentTimeMillis();
			try{
				sub.setCode(cleanCode(parser.parseCodeFromHtml(sub.getCode())));
				VisitedSubmissionSql.insert(sub);
				System.err.println("Parse and insert time used: "
						+ (System.currentTimeMillis() - t) + " ms");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		System.err.println("Main thread stoping...");

		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i].stop = true;
		}
		for (int i = 0; i < HtmlGetter.ThreadNumber; i++) {
			htmlGetter[i].join();
		}
	}

	private static String cleanCode(String code) {
		code = code.replace("&lt;", "<");
		code = code.replace("&gt;", ">");
		code = code.replace("&amp;", "&");
		code = code.replace("&quot;", "\"");
		return code;
	}

	public static void main(String[] args) throws Exception {
		CodeForceSubmissionCrawler instance = new CodeForceSubmissionCrawler();
		instance.runCrawler();
	}
}
